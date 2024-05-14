
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder

def get_data(file_name):

    columns_name = ['status_current_account', 'duration', 'credit_history', 'purpose',
               'credit_amount', 'savings', 'employed_since', 'installment_rate',
               'status_and_sex', 'other_debtors', 'present_residence_since', 'property',
               'age', 'other_installment_plans', 'housing', 'n_credits', 'job',
               'n_maintenance_people', 'telephone', 'foreign', 'Class']
    df = pd.read_csv(file_name, sep=' ', names=columns_name)

    return df

def clean_data_german(df):
    dict_attr = {
        'A11': '< 0',
        'A12': '0 <= X < 200',
        'A13': '>= 200',
        'A14': 'no checking account',
        'A30': 'no credit taken / all paid other banks',
        'A31': 'paid back',
        'A32': 'current credit paid',
        'A33': 'delay',
        'A34': 'critical / other banks credit',
        'A40': 'new car',
        'A41': 'car used',
        'A42': 'furniture',
        'A43': 'radio/tv',
        'A44': 'domestic appliances',
        'A45': 'repairs',
        'A46': 'education',
        'A47': 'vacation',
        'A48': 'retraining',
        'A49': 'business',
        'A410': 'others',
        'A61': '< 100',
        'A62': '100 <= X < 500',
        'A63': '500 <= X < 1000',
        'A64': '>= 1000',
        'A65': 'unknown',
        'A71': 'unemployed',
        'A72': '< 1',
        'A73': '1 <= X < 4',
        'A74': '4 <= X < 7',
        'A75': '>= 7',
        'A91': 'male divorced/separated',
        'A92': 'female divorced/separated/married',
        'A93': 'male single',
        'A94': 'male married/widowed ',
        'A95': 'female single',
        'A101': 'none',
        'A102': 'co-applicant',
        'A103': 'guarantor',
        'A121': 'real estate',
        'A122': 'building society/life insurance',
        'A123': 'car or other',
        'A124': 'unknown / no property',
        'A141': 'bank',
        'A142': 'stores',
        'A143': 'none',
        'A151': 'rent',
        'A152': 'own',
        'A153': 'for free',
        'A171': 'unemployed or unskulled non resident',
        'A172': 'unskilled resident',
        'A173': 'skilled/official',
        'A174': 'management/self-employed/highly qualified/officer',
        'A191': 'none',
        'A192': 'yes',
        'A201': 'yes',
        'A202': 'no'
    }

    df = df.replace(dict_attr)

    df.loc[df['status_and_sex'] != 'female divorced/separated/married', 'status_and_sex'] = 'male'
    df.loc[df['status_and_sex'] == 'female divorced/separated/married', 'status_and_sex'] = 'female'
    df = df.rename(columns={'status_and_sex': 'gender'})

    df.loc[df['Class'] == 1, 'Class'] = 'Accepted'
    df.loc[df['Class'] == 2, 'Class'] = 'Rejected'

    return df

def clean_data_home(df):
    df[df.select_dtypes(include=np.number).columns] = df.select_dtypes(include=np.number).fillna(-1)
    df[df.select_dtypes(exclude=np.number).columns] = df.select_dtypes(exclude=np.number).fillna('nan')
    # df.dropna(axis=1, inplace = True)
    df.rename(columns={'TARGET': 'Class'}, inplace=True)

    return df

def clean_data(df, dataset_name='german'):

    if dataset_name == 'german':
        df = clean_data_german(df)
    elif dataset_name == 'home':
        df = clean_data_home(df)

    return df
    

def process_data_german(df, encoder = 'OneHot'):

    df = df.copy()
    
    df['Class'] = df['Class'].replace('Rejected',0)
    df['Class'] = df['Class'].replace('Accepted',1)
    df['Class'] = df['Class'].astype(int)

    # Create gender column (1 female 0 male)
    if 'status_and_sex' in df.columns:
        df.insert(len(df.columns)-1, 'gender', 
                       np.where(df['status_and_sex'] == 'female divorced/separated/married', 1, 0))
        # Remove status_and_sex column
        df = df.drop('status_and_sex', axis=1)

    
    df['gender'] = df['gender'].replace('male',0)
    df['gender'] = df['gender'].replace('female',1)
    df['gender'] = df['gender'].astype(int)

    # Select cols to encode
    cols_enc = list(df.select_dtypes([object]).columns)

    if encoder == 'OneHot':
        # Encoder creation
        ce_be = ce.OneHotEncoder(cols=cols_enc)
        # transform the data
        df = ce_be.fit_transform(df)

    elif encoder == 'Label':
        label_encoder = LabelEncoder()
        df[cols_enc] = df[cols_enc].apply(label_encoder.fit_transform)

    return df

def process_data_home(df, encoder = 'OneHot'):

    df = df.copy()

    df.rename(columns={'TARGET': 'Class'}, inplace=True)

    # Create gender column (1 female 0 male)
    if 'CODE_GENDER' in df.columns:
        df = df[df['CODE_GENDER'] != 'XNA']

        df.insert(len(df.columns)-1, 'gender', 
                       np.where(df['CODE_GENDER'] == 'F', 1, 0))
        # Remove CODE_GENDER column
        df = df.drop('CODE_GENDER', axis=1)

    # Select cols to encode
    cols_enc = list(df.select_dtypes([object]).columns)

    if encoder == 'OneHot':
        # Encoder creation
        ce_be = ce.OneHotEncoder(cols=cols_enc)
        # transform the data
        df = ce_be.fit_transform(df)

    elif encoder == 'Label':
        label_encoder = LabelEncoder()
        df[cols_enc] = df[cols_enc].apply(label_encoder.fit_transform)

    # Drop nan
    df = df.dropna(axis=1)

    return df

def process_data(df, dataset_name='german', encoder = 'Label'):

    if dataset_name == 'german':
        df = process_data_german(df, encoder)
    elif dataset_name == 'home':
        df = process_data_home(df, encoder)

    return df

def split_data(df, test_size = 0.10, y_name = 'Class', get_test = True, seed = None):

    # Get attributes
    X = df.loc[:, df.columns != y_name]

    # Get class
    y = df[y_name]

    if get_test == True:
    # Stratified division
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = y, random_state=seed)

        return X_train, X_test, y_train, y_test

    else:
        return X, y


def get_df4chi(df, dataset='german'):

    if dataset == 'german':
        df4chi = df.copy()
        df4chi.loc[df4chi['status_and_sex'] != 'female divorced/separated/married', 'status_and_sex'] = 'male'
        df4chi.loc[df4chi['status_and_sex'] == 'female divorced/separated/married', 'status_and_sex'] = 'female'

        df4chi.loc[df4chi['Class'] == 1, 'Class'] = 'Accepted'
        df4chi.loc[df4chi['Class'] == 2, 'Class'] = 'Rejected'

    elif dataset == 'home':
        df4chi = df.copy()
        df4chi.loc[df4chi['CODE_GENDER'] == 'M', 'CODE_GENDER'] = 'male'
        df4chi.loc[df4chi['CODE_GENDER'] == 'F', 'CODE_GENDER'] = 'female'
        df4chi = df4chi[df4chi['CODE_GENDER'] != 'XNA']

        df4chi.loc[df4chi['Class'] == 0, 'Class'] = 'Accepted'
        df4chi.loc[df4chi['Class'] == 1, 'Class'] = 'Rejected'
    
    else:
        return None

    return df4chi

def get_res_df(X, y_true, y_pred):
    
    X = X.reset_index(drop=True)
    y_true = y_true.reset_index(drop=True)
    y_pred = pd.Series(y_pred, name='y_pred').reset_index(drop=True)

    return pd.concat([X, y_true, y_pred], axis=1)

def decoding(df, df_processed, ignore_columns = []):
    joined_df = pd.concat([df, df_processed], axis=1)
    decoding_dict = {}
    
    for column_name in df_processed.columns:
        if column_name not in ignore_columns:
            dec_column_name = ''.join(i for i in column_name if not i.isdigit())[:-1]

            res = joined_df.loc[joined_df[column_name] == 1][dec_column_name].unique()
            decoding_dict[column_name] = res[0]
            
    return decoding_dict

def ignore_attribute_n_values(df, n = 10): # Move to script file
    
    selected_attributes = []
    
    for c in df.columns:
        if is_numeric_dtype(df[c]) or df[c].nunique() < n:
            selected_attributes.append(c)
    
    return df[selected_attributes]

def fill_nan(df):
    df[df.select_dtypes(include=np.number).columns] = df.select_dtypes(include=np.number).fillna(df.select_dtypes(include=np.number).median())
    df[df.select_dtypes(exclude=np.number).columns] = df.select_dtypes(exclude=np.number).fillna('nan_category')
    
    return df





def get_definition(definition_df, col_name):
    return definition_df[definition_df[0] == 'NAME'][definition_df[1] == col_name].iloc[0][4]

def get_category_type(definition_df, col_name):
    return definition_df[definition_df[0] == 'NAME'][definition_df[1] == col_name].iloc[0][2]

def get_values_dict(definition_df, col_name):
    values_df = definition_df[definition_df[0] == 'VAL'][definition_df[1] == col_name]
    values_dict = {} 

    for i,r in values_df.iterrows():
        values_dict[r[4]] = r[6]
        
    return values_dict

def transform_columns_type(df):
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            pass
    
    return df

def process_dictionary(d):
    processed_dict = {}
    
    for k,v in d.items():
        if isinstance(k, str) and k.isdigit():
            k = float(k)
        if isinstance(v, str) and v.isdigit():
            v = float(v)
        processed_dict[k] = v
        
    return processed_dict

def clean_df_census(df, definition_df):

    df_cleaned = transform_columns_type(df)

    for c in df_cleaned.columns:
        if c != 'TARGET' and get_category_type(definition_df, c) == 'C':
            d = process_dictionary(get_values_dict(definition_df, c))
            df_cleaned[c] = df_cleaned[c].replace(d)
        
    return df_cleaned

def calculate_nan_percentage_gender(df, gender_column = 'gender'):
    nan_columns = df.columns[df.isna().any()].tolist() + [gender_column]
    
    # dataframe with only the columns that have NaN values
    df = df[nan_columns]

    # Group the dataframe by gender
    gender_groups = df.groupby(gender_column)
    
    # Calculate the percentage of NaN values per column for each gender group
    nan_percentages = gender_groups.apply(lambda x: x.isna().mean() * 100)
    
    # Add difference
    nan_percentages.loc['Difference'] = abs(nan_percentages.diff().iloc[-1])
    
    # Print the result
    return nan_percentages

def clean_nan(df, percentage = 10, verbose = False, gender_column = 'gender'):
    # Calculate percentage of NaNs per column
    nan_percentages = df.isna().sum() / len(df) * 100
    
    # Print NaN percentages per column
    if verbose == True:
        print('NaN percentages per column:\n', nan_percentages[nan_percentages > 0])
    
    # Get column names where more than percentage% of values are NaN
    nan_columns = nan_percentages[nan_percentages > percentage].index
    if verbose == True:
        print('Removing:\n', nan_columns)
    
    # Remove columns with more than percentage% NaN values
    df = df.drop(nan_columns, axis=1)
    
    # Remove columns where nan% difference between males and females is greater than 5%
    gender_nan_perc = calculate_nan_percentage_gender(df, gender_column)
    nan_gender_columns = gender_nan_perc.loc[:, gender_nan_perc.iloc[2, :] > 5].columns
    df = df.drop(nan_gender_columns, axis=1)
    
    # Fill NaN values in numerical columns with median
    numerical_columns = df.select_dtypes(include=['number']).columns
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median()) 
    
    # Replace NaN values in categorical columns with 'nan_category'
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns   
    df[categorical_columns] = df[categorical_columns].fillna('nan_category')
    
    return df

def process_data_census(df):

    df = df.copy()

    # Create gender column (1 female 0 male)
    if 'SEX' in df.columns:
        df.insert(len(df.columns)-1, 'gender', np.where(df['SEX'] == 'Female', 1, 0))
        df = df.drop('SEX', axis=1)
        
    if 'TARGET' in df.columns:
        df.insert(len(df.columns)-1, 'Class', np.where(df['TARGET'] == True, 1, 0))
        df = df.drop('TARGET', axis=1)
    
    # Remove attributes related with income 
    attributes_to_remove = ['PINCP', 'PERNP', 'WAGP', 'SEMP', 'RETP', 'INTP',
                            'SSP', 'SSIP', 'OIP', 'SERIALNO']
    df = df.drop(attributes_to_remove,axis=1, errors='ignore')
    
    # Remove Person Weight columns (columns that start with "PWGTP")
    df = df.filter(regex='^(?!PWGTP)')
    
    # Nan dealing
    df = clean_nan(df)
    
    # Encoding (factorize)
    cols_enc = list(df.select_dtypes([object]).columns)

    for col in cols_enc:
        df[col], _ = pd.factorize(df[col])

    return df
