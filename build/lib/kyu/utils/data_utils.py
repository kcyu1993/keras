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


def csv_saver(filepath, entries, title=None, **kwargs):
    """
    Write the CSV

    Parameters
    ----------
    filepath
    entries
    title
    kwargs

    Returns
    -------

    """
    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter=',', **kwargs)
        if title:
            writer.writerow(title)
        for e in entries:
            writer.writerow(e)
    return filepath
