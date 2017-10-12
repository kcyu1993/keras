"""
Define image loading related operations

"""
import csv

def csv_reader(filepath, **kwargs):
    """

    Parameters
    ----------
    filepath
    kwargs : passed into csv reader

    Returns
    -------

    """
    # title = []
    entry = []
    with open(filepath, 'rb') as f:
        reader = csv.reader(f, **kwargs)
        title = reader.next()
        for row in reader:
            entry.append(row)

    # Interesting
    return title, entry
