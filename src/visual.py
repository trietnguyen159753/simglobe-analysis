import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import os


def visualize_rsquared(df: pl.LazyFrame):
    for country in df.select("country").unique().collect().to_series():
        df_filter = df.filter(
            pl.col("country") == country,
            pl.col("variable") == "r_squared",
        ).sort("output_var")

        _, ax = plt.subplots(figsize=(7.5, 5))

        sns.lineplot(
            data=df_filter.collect(),
            x="period",
            y="value",
            hue="output_var",
            ax=ax,
            palette="icefire",
            errorbar=None,
        )

        handles, labels = ax.get_legend_handles_labels()

        ax.legend(handles=handles, labels=labels, loc="best")

        plt.title(f"R-squared values for {country}")
        plt.xlabel("Period")
        plt.ylabel("R-squared")
        plt.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()

        output_dir = "./visual/r-squared/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(
            os.path.join(output_dir, f"{country}.png"), dpi=100, bbox_inches="tight"
        )
        plt.close()

        print(f"Done plotting {country} for r_squared")


def visualize_coef_output_var(
    df: pl.LazyFrame,
    input_vars: list[str],
    output_vars: list[str],
):
    for country in df.select("country").unique().collect().to_series():
        for var in output_vars:
            
            input_var_coef = [f"{var}_coef" for var in input_vars]
            
            df_filter = df.filter(
                pl.col("country") == country,
                pl.col("output_var") == var,
                pl.col("variable").is_in(input_var_coef),
            ).sort("period")
            
            fig, ax = plt.subplots(figsize=(8.5, 5))

            sns.barplot(
                data=df_filter.collect(),
                x="period",
                y="value",
                hue="variable",
                palette="icefire",
                errorbar=None
            )

            handles, labels = ax.get_legend_handles_labels()

            ax.legend(
                handles=handles,
                labels=labels,
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
            )

            plt.title(f"Coefficient for regression on {var} of {country}")
            plt.xlabel("Period")
            plt.ylabel(var)
            plt.grid(linestyle="--", alpha=0.3)
            plt.tight_layout()

            output_dir = f"./visual/{var}/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            plt.savefig(
                os.path.join(output_dir, f"{country}.png"), dpi=100, bbox_inches="tight"
            )
            plt.close()

            print(f"Done plotting coef of {var} for {country}")
