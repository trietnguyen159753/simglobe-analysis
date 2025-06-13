from statsmodels.regression.linear_model import RegressionResultsWrapper
import polars as pl
import pandas as pd
import statsmodels.api as sm



def permutation(df: pl.LazyFrame, cols: list[str]) -> pl.DataFrame:
    return df.select(cols).unique().collect()


def filter_outlier(
    df: pl.LazyFrame,
    filter_enable: bool,
    iqr_threshold: int,
    output_var: list[str],
) -> pl.LazyFrame:
    print("Filtering outliers...")

    bound = []

    if filter_enable:
        for var in output_var:
            q1 = pl.col(var).quantile(0.25)
            q3 = pl.col(var).quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - iqr_threshold * iqr
            upper_bound = q3 + iqr_threshold * iqr

            bound.append(lower_bound.alias(f"lower_{var}"))
            bound.append(upper_bound.alias(f"upper_{var}"))

        df_temp = df.with_columns(bound)

        filter_expr = [
            pl.col(var).is_between(pl.col(f"lower_{var}"), pl.col(f"upper_{var}"))
            for var in output_var
        ]

        combined_filter = pl.all_horizontal(*filter_expr)

    df_final = df_temp.filter(combined_filter).drop(pl.col("^.*er_.*$"))

    pre_outlier_count = df.collect().shape[0]
    post_outlier_count = df_final.collect().shape[0]

    print(
        f"Filtering outliers done, removing {pre_outlier_count - post_outlier_count} entries from {pre_outlier_count} to {post_outlier_count}"
    )

    return df_final


def filter_influential(
    df: pl.LazyFrame,
    output_vars: list[str],
    input_vars: list[str],
) -> tuple[pl.LazyFrame, dict[str, RegressionResultsWrapper]]:
    print("Filtering influential points...")

    x: pd.DataFrame = df.select(input_vars).collect().to_pandas()
    output_vars_pd: pd.DataFrame = df.select(output_vars).collect().to_pandas()

    cooks_distance: dict[str, list[float]] = {}
    model_final: dict[str, RegressionResultsWrapper] = {}

    for var in output_vars_pd:
        y = output_vars_pd[[var]]
        x = sm.add_constant(x)
        model: RegressionResultsWrapper = sm.OLS(y, x).fit()
        model_final[var] = model
        cooks_distance[var] = model.get_influence().cooks_distance[0]

    new_cols_expr = [
        pl.Series(f"cooks_{output_var}", distance).cast(pl.Float32)
        for output_var, distance in cooks_distance.items()
    ]

    df_add_cooks = df.with_columns(new_cols_expr)

    cutoff = 4 / df.collect().shape[0]

    filter_expr = [
        pl.col(f"cooks_{output_var}") <= cutoff for output_var in output_vars
    ]

    df_final = df_add_cooks.filter(filter_expr).drop(pl.col("^cooks.*$"))

    pre_outlier_count = df.collect().shape[0]
    post_outlier_count = df_final.collect().shape[0]

    print(
        f"Filtering influential point done, removing {pre_outlier_count - post_outlier_count} entries from {pre_outlier_count} to {post_outlier_count}"
    )

    return df_final, model_final


def format_long_df(df: pl.LazyFrame) -> pl.LazyFrame:
    df_long = (
        df.with_columns(
            pl.when(pl.col("r_squared") < 0)
            .then(0)
            .otherwise(pl.col("r_squared"))
            .alias("r_squared")
        )
        .unpivot(
            df.drop("country", "period", "scenario", "output_var")
            .collect_schema()
            .names(),
            index=["country", "period", "scenario", "output_var"],
        )
        .with_columns(pl.col("country").str.to_titlecase())
    )

    return df_long
