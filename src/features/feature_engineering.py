
import pandas as pd


def create_features(df, prediction_mode="later"):
    """
    Create features for school student performance analysis.

    prediction_mode:
        early - Excludes G1 and G2.
        later - Uses G1 and G2.
    """

    df = df.copy()

    df["parent_education_avg"] = (
        df["Medu"] + df["Fedu"]
    ) / 2

    df["total_alcohol"] = (
        df["Dalc"] + df["Walc"]
    )

    if prediction_mode == "early":
        df = df.drop(
            columns=["G1", "G2"],
            errors="ignore"
        )

    elif prediction_mode == "later":
        df["total_grade"] = df["G1"] + df["G2"]
        df["grade_progress"] = df["G2"] - df["G1"]

    else:
        raise ValueError(
            "prediction_mode must be 'early' or 'later'"
        )

    return df
