import pandas as pd


def clean_dataset(df):

    df = df.copy()

    df = df.dropna(
        subset=["claim", "label"]
    )

    df["claim"] = (
        df["claim"]
        .astype(str)
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    valid_labels = {
        "false",
        "mixture",
        "true",
        "unproven"
    }

    df = df[
        df["label"].isin(valid_labels)
    ]

    df = df[
        df["claim"].str.len() > 0
    ]

    return df


def test_clean_dataset():

    sample = pd.DataFrame({
        "claim": [
            "A health claim",
            None,
            "   ",
            "Another claim"
        ],

        "label": [
            "true",
            "false",
            "unproven",
            "invalid"
        ]
    })

    cleaned = clean_dataset(
        sample
    )

    assert len(cleaned) == 1

    assert cleaned.iloc[0]["claim"] == (
        "A health claim"
    )

    assert cleaned.iloc[0]["label"] == (
        "true"
    )