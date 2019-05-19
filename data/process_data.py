#importing libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """load the datas from given csv files and retuns 
    merged dataframe
    Parameters
    *********
    messages_filepath : string
        location of the messages csv file
    categories_filepath : string
        location of the categories csv file
    Returns
    ********
    pandas.DataFrame
        The dataframe with merged both the 
        csv input data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """
    Process a dataframe and perform the all cleaning 
    operations by dropping unnessesary columns
    Parameters:
    *******************
    df: pandas.DataFrame
        The pandas.Dataframe to be processed
    Returns:
    **********************
    pandas.DataFrame
        The processed dataframe which had cleaning in this function
    """

    # THis code will create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', n=None, expand=True)

    # use the first row to extract a list of new column names for categories.
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    
    category_colnames = categories.iloc[0].str[:-2]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        # and chnage the datatype to int
        # convert column from string to numeric
        categories[column] = categories[column].str[-1].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns='categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates(keep='first')

    return df


def save_data(df, database_filename):
    """
    saves the df dataframe to a Sql-lite Database
    Parameters:
    ***************************
    df: pandas.DataFrame
        The pandas.Dataframe to be saved in Sql
    database_filename: string
        The filename path for the database
    Returns
    ********************
    None
    """
    print('Saving dataframe {} to {} Sql-lite database: '.format(df, database_filename))
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages_categories', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()