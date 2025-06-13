import polars as pl

from .config import config

from .util import permutation
from .process import (
    load_process,
    filter_process,
    regression_result_process,
    visualize_process,
)


def main():
    df = load_process(config.scenario)

    df_summaries = []

    for unique_group in permutation(df, config.unique).iter_rows():
        _, model = filter_process(
            df,
            unique_group,
            config.filter_param.filter_enable,
            config.filter_param.iqr_threshold,
            config.input_var,
            config.output_var,
        )

        df_summary = regression_result_process(
            model,
            unique_group,
            config.input_var,
        )

        df_summaries.append(df_summary)

    df_visual = pl.concat(df_summaries)
    
    df_visual = pl.scan_csv("./data/regression.csv")

    visualize_process(df_visual, config.input_var, config.output_var)


if __name__ == "__main__":
    main()
