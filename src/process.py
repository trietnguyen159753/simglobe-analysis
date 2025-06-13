import polars as pl
from .util import format_long_df, filter_outlier, filter_influential
from .visual import visualize_rsquared, visualize_coef_output_var
from statsmodels.regression.linear_model import RegressionResultsWrapper


def load_process(scenarios: list[str]) -> pl.LazyFrame:
    print("Importing data...")

    df: list[pl.LazyFrame] = [
        pl.scan_parquet(f"./data/{scenario_unit}.parquet")
        .with_columns(scenario=pl.lit(scenario_unit))
        .with_columns(pl.col(pl.Int32).cast(pl.UInt8))
        for scenario_unit in scenarios
    ]

    return pl.concat(df)


def filter_process(
    df: pl.LazyFrame,
    permutation: tuple[str, int, str],
    filter_enable: bool,
    iqr_threshold: int,
    input_vars: list[str],
    output_vars: list[str],
) -> tuple[pl.LazyFrame, dict[str, RegressionResultsWrapper]]:
    country, period, scenario = permutation

    df_unique = df.filter(
        pl.col("country") == country,
        pl.col("period") == period,
        pl.col("scenario") == scenario,
    )

    df_outlier = filter_outlier(
        df_unique,
        filter_enable,
        iqr_threshold,
        output_vars,
    )

    df_influential, model = filter_influential(
        df_outlier,
        output_vars,
        input_vars,
    )

    return df_influential, model


def regression_result_process(
    models: dict[str, RegressionResultsWrapper],
    permutation: tuple[str, int, str],
    input_vars: list[str],
) -> pl.LazyFrame:
    country, period, scenario = permutation

    results = []

    for output_var, model in models.items():
        result: dict[str, any] = {
            "country": country,
            "period": period,
            "scenario": scenario,
            "output_var": output_var,
            "n_rows": model.nobs,  # Use the model's nobs for accuracy
            "r_squared": model.rsquared,
            "prob_f_stat": 0.0 if model.f_pvalue <= 1e-4 else model.f_pvalue,
            # "intercept_coef": model.params["const"],
        }

        for i, input_var in enumerate(input_vars):
            idx = i + 1  # +1 to account for constant
            result[f"{input_var}_coef"] = model.params.iloc[idx]
            result[f"{input_var}_pvalue"] = (
                0.0 if model.pvalues.iloc[idx] <= 1e-4 else model.pvalues.iloc[idx]
            )

        results.append(result)

    return pl.LazyFrame(results)


def visualize_process(
    df: pl.LazyFrame,
    input_vars: list[str],
    output_vars: list[str],
):
    df_long = format_long_df(df)

    visualize_rsquared(df_long)

    visualize_coef_output_var(df_long, input_vars, output_vars)

    # ### Caching

    # print("Caching...")

    # os.makedirs("cache", exist_ok=True)
    # df_filter.write_parquet("./cache/df_filter.parquet")
    # with open(os.path.join("cache", "cooks_distance.pkl"), "wb") as f:
    #     pickle.dump(cooks_distance, f)

    # print("Caching done.")

    # df_filter = pl.read_parquet("./cache/df_filter.parquet")
    # with open("./cache/cooks_distance.pkl", "rb") as f:
    #     cooks_distance = pickle.load(f)
