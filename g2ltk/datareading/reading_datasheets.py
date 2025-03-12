from typing import Optional, Any, Tuple, Dict, List, Union
import os # to navigate in the directories
import numpy as np
import pandas as pd

from .. import utility, logging

from . import generate_dataset_path


__DATASHEET_NAME__ = 'datasheet'

def generate_datasheet_path(dataset:str) -> Optional[str]:
    dataset_path = generate_dataset_path(dataset)

    ext = '.xlsx'
    xlsx_spreadsheets = [f[:-len(ext)] for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith(ext)]
    xlsx_spreadsheets.sort()

    ext = '.ods'
    ods_spreadsheets = [f[:-len(ext)] for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and f.endswith(ext)]
    ods_spreadsheets.sort()

    datasheet_name = None

    # First, we try the default name
    global __DATASHEET_NAME__
    if os.path.isfile(os.path.join(dataset_path, __DATASHEET_NAME__ + '.xlsx')):
        datasheet_name = __DATASHEET_NAME__ + '.xlsx'
    elif os.path.isfile(os.path.join(dataset_path, __DATASHEET_NAME__ + '.ods')):
        logging.log_warning('Old format .ods used, consider upgrading to .xlsx')
        datasheet_name = __DATASHEET_NAME__ + '.ods'
    elif len(xlsx_spreadsheets) == 1 and len(ods_spreadsheets) == 0:
        datasheet_name = xlsx_spreadsheets[0] + '.xlsx'
        logging.log_warning(f'Name of datasheet is "{datasheet_name}", consider changing to "{__DATASHEET_NAME__ + ".xlsx"}" for consistency')
    elif len(xlsx_spreadsheets) == 0 and len(ods_spreadsheets) == 1:
        datasheet_name = ods_spreadsheets[0] + '.ods'
        logging.log_warning(f'Name of datasheet is "{datasheet_name}", consider changing to "{__DATASHEET_NAME__ + ".xlsx"}" for consistency')
    elif len(xlsx_spreadsheets) == 0 and len(ods_spreadsheets) == 0:
        logging.log_warning(f'No datasheet in dataset {dataset}.')
    else:
        logging.log_warning(f'To many datasheet in dataset {dataset}: {xlsx_spreadsheets+ods_spreadsheets}.')
    if datasheet_name is None:
        return None
    datasheet_path = os.path.join(dataset_path, datasheet_name)
    return datasheet_path

def obtain_metainfo(dataset:str):
    datasheet_path = generate_datasheet_path(dataset)
    if datasheet_path is None:
        return None

    file = pd.ExcelFile(datasheet_path)
    if 'metainfo' in file.sheet_names:
        metainfo_sheet_name = 'metainfo'
    else:
        metainfo_sheet_name = file.sheet_names[0]
        logging.log_warning(f'Name of the metainfo sheet is {metainfo_sheet_name}, consider changing to "{"metainfo"}" for consistency.')

    metainfo = pd.read_excel(datasheet_path, sheet_name=metainfo_sheet_name, skiprows=2)
    return metainfo
