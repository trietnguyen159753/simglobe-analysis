import polars as pl
from sklearn.linear_model import LinearRegression
from .config import config
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_process(scenarios: list[str]) -> pl.LazyFrame:
    print("Importing data...")

    df: list[pl.LazyFrame] = [
        pl.scan_parquet(f"./data/{scenario_unit}.parquet")
        .with_columns(scenario=pl.lit(scenario_unit))
        .with_columns(pl.col(pl.Int32).cast(pl.UInt8))
        .with_columns(country=pl.col("country").str.to_titlecase())
        for scenario_unit in scenarios
    ]

    return pl.concat(df)


def filter_process(
    df: pl.LazyFrame,
    output_vars: list[str],
) -> pl.LazyFrame:
    
    print("Filtering outlier...")

    filter_expr = [
        pl.col(var).is_between(
            pl.col(var).quantile(config.trim_radius).over(config.unique),
            pl.col(var).quantile(1 - config.trim_radius).over(config.unique),
        )
        for var in output_vars
    ]

    combined_filter = pl.all_horizontal(*filter_expr)
    df_filter = df.filter(combined_filter)
    print(
        f"Filtering complete. Removing {df.collect().shape[0] - df_filter.collect().shape[0]} entries, from {df.collect().shape[0]} to {df_filter.collect().shape[0]}"
    )

    return df_filter


def eda_process(df: pl.LazyFrame) -> None:
    
    for country in df.select("country").unique().collect().to_series():
            print(f"Visualizing {country}...")
            for var in config.output_var:
                
                df_filter = df.filter(
                    pl.col("country") == country,
                ).select("country", "period", "scenario", var).collect().sample(fraction = 0.1)

                name = f"{var} - {country}"

                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax = sns.violinplot(
                    data=df_filter,
                    x="period",
                    y=var,
                    palette=sns.color_palette("light:#5A9", 2),
                    hue="scenario",
                    split=True,
                    inner="box",
                    cut=0,
                    density_norm="width"
                )

                plt.title(name)
                plt.tight_layout()
                if var == "approval_index":
                    plt.ylim(0, 100)
                    plt.yticks([0, 25, 50, 75, 100])
                    plt.grid(axis="y", linestyle="--", alpha=0.4)

                plt.savefig(f"./visual/eda/{name}.png", dpi=80)
                plt.close()
                
    for period in df.select("period").unique().collect().to_series():
            print(f"Visualizing period {period}...")
            for var in config.output_var:
                
                df_filter = df.filter(
                    pl.col("period") == period,
                ).select("country", "period", "scenario", var).collect().sample(fraction = 0.1).sort("country")

                name = f"{var} - period {period}"

                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax = sns.violinplot(
                    data=df_filter,
                    x="country",
                    y=var,
                    palette=sns.color_palette("light:#5A9", 2),
                    hue="scenario",
                    split=True,
                    inner="box",
                    cut=0,
                    density_norm="width"
                )

                plt.title(name)
                plt.tight_layout()
                if var == "approval_index":
                    plt.ylim(0, 100)
                    plt.yticks([0, 25, 50, 75, 100])
                    plt.grid(axis="y", linestyle="--", alpha=0.4)

                plt.savefig(f"./visual/eda/{name}.png", dpi=80)
                plt.close()
                
                
def regression_process(
    df: pl.LazyFrame,
) -> pl.LazyFrame:
    print("Running regression...")
    
    regression = []
    
    for country, period, scenario in df.select(config.unique).unique().collect().iter_rows():
        print(country, period, scenario)
        
        df_unique = df.filter(
            pl.col("country") == country,
            pl.col("period") == period,
            pl.col("scenario") == scenario,
        )
        
        X = df_unique.select(config.input_var).collect().to_numpy()
        
        for dep_var in config.output_var:
            y = df_unique.select(dep_var).collect().to_numpy()
            
            model = LinearRegression()
            model.fit(X, y)
            
            coeffs = {f"{ind_var}": coef for ind_var, coef in zip(config.input_var, model.coef_.ravel())}
            results = {
                'country': country,
                'period': period,
                'scenario': scenario,
                'dep_var': dep_var,
                'r_squared': model.score(X, y),
                'intercept': model.intercept_
            }
            
            regression.append(results | coeffs)
            
    return pl.LazyFrame(regression)
            

def visualize_process(
    df: pl.LazyFrame,
):
    
    df_long = (
        df.with_columns(
            pl.when(pl.col("r_squared") < 0)
            .then(0)
            .otherwise(pl.col("r_squared"))
            .alias("r_squared")
        )
        .unpivot(
            df.drop("country", "period", "scenario", "dep_var")
            .collect_schema()
            .names(),
            index=["country", "period", "scenario", "dep_var"],
        )
        .with_columns(pl.col("country").str.to_titlecase())
    )
    
    for country, scenario in df.select("country", "scenario").unique().collect().iter_rows():
        
        print(f"Plotting {country}, {scenario}...")
        
        df_unique = df_long.filter(
            pl.col("country") == country,
            pl.col("scenario") == scenario,
        )
        
        df_rsquared = df_unique.filter(pl.col("variable") == "r_squared").sort("dep_var")
        
        name = f"r-squared - {country} - {scenario}"
        
        _, ax = plt.subplots(figsize=(7.5, 5))
        sns.lineplot(
            data=df_rsquared.collect(),
            x="period",
            y="value",
            hue="dep_var",
            ax=ax,
            palette="icefire",
            errorbar=None,
        )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc="best")
        plt.title(name)
        plt.xlabel("Period")
        plt.ylabel("R-squared")
        plt.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()

        output_dir = "./visual/regression/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(f"{output_dir}{name}.png", dpi=100, bbox_inches="tight")
        plt.close()
        
        for dep_var in config.output_var:
        
            df_coef = df_unique.filter(
                pl.col("dep_var") == dep_var,
                pl.col("variable").is_in(config.input_var)
            )
            
            name = f"{dep_var} - {country} - {scenario}"
        
            _, ax = plt.subplots(figsize=(8.5, 5))
            sns.barplot(
                data=df_coef.collect(),
                x="period",
                y="value",
                hue="variable",
                palette="icefire",
                errorbar=None,
            )

            handles, labels = ax.get_legend_handles_labels()

            ax.legend(
                handles=handles,
                labels=labels,
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
            )

            plt.title(name)
            plt.xlabel("Period")
            plt.ylabel(dep_var)
            plt.grid(linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}{name}.png", dpi=100, bbox_inches="tight")
            plt.close()


