import pandas as pd

def compare_excels(path1, path2):
    df1 = pd.read_excel(path1)
    df2 = pd.read_excel(path2)

    if df1.shape != df2.shape:
        print(f"Different shapes: {df1.shape} vs {df2.shape}")
        return False

    if list(df1.columns) != list(df2.columns):
        print("Different columns")
        print(f"  File1: {list(df1.columns)}")
        print(f"  File2: {list(df2.columns)}")
        return False

    diff = (df1 != df2) & ~(df1.isna() & df2.isna())
    n_diff = diff.sum().sum()

    if n_diff == 0:
        print("Files are identical")
        return True
    else:
        print(f"Files differ in {n_diff} cell(s):")
        rows, cols = diff.any(axis=1), diff.any(axis=0)
        print(df1.loc[rows, cols].compare(df2.loc[rows, cols]))
        return False


if __name__ == "__main__":
    path1 = "../../data/processed/others-ignore/processed_20260219_000215.xlsx"
    path2 = "../../data/processed/others-ignore/processed_20260319_192303.xlsx"
    compare_excels(path1, path2)
