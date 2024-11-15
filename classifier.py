import click
import pandas as pd
from utils.ml import load_models, categorize


def parse_natwest_csv(file_path):
    def check_payment(txt):
        return 'AMERICAN EXP 3773' in txt.upper()

    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    mask = data['Description'].apply(check_payment)
    data = data[~mask]
    data['Card'] = 'NATWEST'
    return data[['Date', 'Description', 'Value', 'Card']].rename({'Value': 'Amount'}, axis=1)


def parse_amex_xlsx(file_path):
    def check_payment(txt):
        return 'PAYMENT RECEIVED' in txt.upper()

    df= pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Amount'] = df['Amount'] * -1
    mask = (df['Amount'] > 0) & df['Description'].apply(check_payment)
    df = df[~mask]
    df['Card'] = 'AMEX'
    return df[['Date', 'Description', 'Amount', 'Card']]


@click.command()
@click.option('--natwest', type=click.File('r'), required=False, help="Path to NatWest file (optional).")
@click.option('--amex', type=click.File('r'), required=False, help="Path to Amex file (optional).")
def process_files(natwest, amex):
    """Process NatWest and Amex files if provided."""
    natwest_content = None
    amex_content = None
    data = []

    if natwest:
        click.echo("Processing NatWest file...")
        natwest_content = parse_natwest_csv(natwest)
    else:
        click.echo("NatWest file not provided.")

    if amex:
        click.echo("Processing Amex file...")
        amex_content = parse_amex_xlsx(amex)
    else:
        click.echo("Amex file not provided.")

    if natwest_content is not None:
        data.append(natwest_content)

    if amex_content is not None:
        data.append(amex_content)

    if len(data) == 0:
        click.echo('You must pass at least one file')
        exit(-1)

    data = pd.concat([natwest_content, amex_content])

    ml, le = load_models()
    data['Category'] = categorize(data, model=ml, le=le)
    print(data.set_index('Date'))


if __name__ == '__main__':
    process_files()