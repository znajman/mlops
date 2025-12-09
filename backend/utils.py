import pandas as pd

def create_new_features(df, col_name):
    """Create new features based on the description in hourse-colic.names.original
    
    Keyword arguments:
    df -- DataFrame with the data
    col_name -- name of the column to be modified

    Returns:
        df -- DataFrame with the new features
    """
    modified_col_name = f"{col_name}_modified"
    df[modified_col_name] = df[col_name].apply(lambda x: str(x)
                                        if pd.notna(x) and len(str(x)) > 3
                                        else None)

    # Create new features based on the description in hourse-colic.names.original
    df[f"{col_name}_lesion_site"] = df[modified_col_name].apply(lambda x: int(x[:2])
                                                if x and x[:2] in ['00', '11']
                                                else int(x[0])
                                                if x else None)
    df[f"{col_name}_type"] = df[modified_col_name].apply(lambda x: int(x[2])
                                            if x and x[:2] in ['00', '11']
                                            else int(x[1])
                                            if x else None)
    df[f"{col_name}_subtype"] = df[modified_col_name].apply(lambda x: int(x[3])
                                                if x and x[:2] in ['00', '11']
                                            else int(x[2])
                                            if x else None)
    df[f"{col_name}_specific_code"] = df[modified_col_name].apply(lambda x: int(x[4:])
                                                    if x and len(x) > 4 and x[3:] != '10'
                                                    else int(x[3:])
                                                    if x and len(x) > 3
                                                    else None)

    df = df.drop(columns=[modified_col_name])
    return df