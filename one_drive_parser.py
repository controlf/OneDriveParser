'''
MIT License

Control-F - Microsoft OneDrive Parser

Copyright (c) 2022 Control-F Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

    Author: mike.bangham@controlf.co.uk
    Python3 - Tested Android (iOS to come)
    Parse the OneDrive container/sandbox (SQLite and Stream Cache databases/file tree)
    Example usage: python com.microsoft.skydrive.tar '.'
    Requirements: pillow, numpy, pandas, python-opencv, filetype

'''

__version__ = 0.01
__author__ = 'mike.bangham@controlf.co.uk'
__description__ = 'Control-F - Microsoft OneDrive Parser'

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from os.path import join as pj
from os.path import isfile, dirname, basename, isdir, abspath, normpath, relpath
import zipfile
import tarfile
import webbrowser as wb
import filetype
import PIL.Image
import base64
import cv2
import json
from io import BytesIO
from distutils.dir_util import copy_tree

# so we can update slices of the dataframe
pd.options.mode.chained_assignment = None


def timestamp_converter(df, column_header, ts_format):
    if ts_format == 's':
        df[column_header] = pd.to_datetime(abs(df[column_header]), unit='s', errors='ignore')
    elif ts_format == 'ms':
        df[column_header] = pd.to_datetime(abs(df[column_header]), unit='ms', errors='ignore')
    elif ts_format == 'ns':
        df[column_header] = pd.to_datetime(abs(df[column_header]), unit='ns', errors='ignore')
    df[column_header] = df[column_header].dt.strftime('%d-%m-%Y %H:%M:%S')
    return df


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = pj(abspath("."), 'res')
    return pj(base_path, relative_path)


report_components = resource_path('html_report')

file_headers = {b'\xFF\xD8\xFF': ['jpeg', 'image'],
                b'\x89PNG\x0D\x0A\x1A\x0A': ['png''image'],
                b'GIF': ['gif', 'image'],
                b'BM': ['bmp', 'image'],
                b'\x00\x00\x01\x00': ['ico', 'image'],
                b'\x49\x49\x2A\x00': ['tif', 'image'],
                b'\x4D\x4D\x00\x2A': ['tif', 'image'],
                b'RIFF': ['avi', 'video'],
                b'OggS\x00\x02': ['ogg', 'video'],
                b'ftypf4v\x20': ['f4v', 'video'],
                b'ftypF4V\x20': ['f4v', 'video'],
                b'ftypmmp4': ['3gp', 'video'],
                b'ftyp3g2a': ['3g2', 'video'],
                b'matroska': ['mkv', 'video'],
                b'\x01\x42\xF7\x81\x01\x42\xF2\x81)': ['mkv', 'video'],
                b'moov': ['mov', 'video'],
                b'skip': ['mov', 'video'],
                b'mdat': ['mov', 'video'],
                b'\x00\x00\x00\x14pnot': ['mov', 'video'],
                b'\x00\x00\x00\x08wide)': ['mov', 'video'],
                b'ftypmp41': ['mp4', 'video'],
                b'ftypavc1': ['mp4', 'video'],
                b'ftypMSNV': ['mp4', 'video'],
                b'ftypFACE': ['mp4', 'video'],
                b'ftypmobi': ['mp4', 'video'],
                b'ftypmp42': ['mp4', 'video'],
                b'ftypMP42': ['mp4', 'video'],
                b'ftypdash': ['mp4', 'video'],
                b'\x30\x26\xB2\x75\x8E\x66\xCF\x11\xA6\xD9\x00\xAA\x00\x62\xCE\x6C': ['wmv', 'video'],
                b'4XMVLIST': ['4xm', 'video'],
                b'FLV\x01': ['flv', 'video'],
                b'\x1A\x45\xDF\xA3\x01\x00\x00\x00': ['webm', 'video']}


def get_image_type(img_fp):
    file_typ, file_ext = None, None
    kind = filetype.guess(img_fp)
    if kind is None:
        with open(img_fp, 'rb') as bf:
            line = bf.read(50)
            for head, ext in file_headers.items():
                if head in line:
                    file_typ, file_ext = ext
    else:
        file_typ, file_ext = kind.mime, kind.extension
    return file_typ, file_ext


def generate_thumbnail(fp, thmbsize=64):
    file_type, file_ext = get_image_type(fp)
    if file_type and file_ext:
        if file_type.startswith('image'):
            img = PIL.Image.open(fp, 'r')
        elif file_type.startswith('video'):
            try:
                cap = cv2.VideoCapture(fp)
                _, cv2_img = cap.read()
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(cv2_img)
                file_ext = 'JPEG'
            except:
                img = PIL.Image.open(resource_path('blank_jpeg.png'), 'r')
                file_ext = 'PNG'
        else:
            img = PIL.Image.open(resource_path('blank_jpeg.png'), 'r')
            file_ext = 'PNG'
    else:
        img = PIL.Image.open(resource_path('blank_jpeg.png'), 'r')
        file_ext = 'PNG'
    if file_ext == 'jpg':
        file_ext = 'jpeg'

    hpercent = (thmbsize / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((wsize, thmbsize), PIL.Image.ANTIALIAS)

    buf = BytesIO()
    img.save(buf, format=file_ext.upper())
    b64_thumb = base64.b64encode(buf.getvalue()).decode('utf8')
    return b64_thumb


class NpEncoder(json.JSONEncoder):
    # converts numpy objects so they can be serialised
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class OneDriveParser:
    def __init__(self, archive, output_dir):
        self.archive = archive
        self.save_dir = pj(output_dir, 'report_files')
        required = ['QTMetadata.db', 'streamcache']
        key_dir = 'com.microsoft.skydrive'
        files_dict = self.extract_archive(required, key_dir)
        if files_dict['db'] and files_dict['cache']:
            table_dict = self.parse(files_dict)
            if table_dict:
                mr = MakeReport(self.save_dir, table_dict['merged'], table_dict['metadata'], True)
                mr.build()

    def extract_archive(self, required, key_dir):
        # extracts the files we need whilst maintaining folder structure
        files_dict = dict(db=None, cache=list())
        os.makedirs(self.save_dir, exist_ok=True)
        if zipfile.is_zipfile(self.archive):
            with zipfile.ZipFile(self.archive, 'r') as zip_obj:
                archive_members = zip_obj.namelist()
                for archive_member in archive_members:
                    if key_dir in archive_member and any(sub_path in archive_member for sub_path in required):
                        if archive_member.endswith('/'):
                            os.makedirs(self.save_dir + '/' + archive_member, exist_ok=True)
                        else:
                            file = abspath(self.save_dir + '/{}'.format(archive_member))
                            if not isdir(dirname(file)):
                                os.makedirs(dirname(file))
                            with open(file, 'wb') as file_out:
                                zip_fmem = zip_obj.read(archive_member)
                                file_out.write(zip_fmem)
                            if file.endswith('.db'):
                                files_dict['db'] = file
                            else:
                                files_dict['cache'].append(file)
        else:
            with tarfile.open(self.archive, 'r') as tar_obj:
                for member in tar_obj:
                    if key_dir in member.name and any(sub_path in member.name for sub_path in required):
                        if member.isdir():
                            os.makedirs(self.save_dir + '/' + member.name.replace(':', '').replace('%', '_'), exist_ok=True)
                        else:
                            file = abspath(self.save_dir+'/{}'.format(member.name.replace(':', '').replace('%', '_')))
                            if not isdir(dirname(file)):
                                os.makedirs(dirname(file))
                            with open(file, 'wb') as file_out:
                                tar_fmem = tar_obj.extractfile(member)
                                file_out.write(tar_fmem.read())
                            if file.endswith('.db'):
                                files_dict['db'] = file
                            else:
                                files_dict['cache'].append(file)

        return files_dict

    def parse(self, files_dict):
        conn = sqlite3.connect(files_dict['db'])
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

        # Generate a dataframe for each required table in the database and store it in a dictionary
        table_dict = {'stream_cache': {'updates': [{'stream_cache_streamType': {1: 'Original', 2: 'Thumbnail',
                                                                                4: 'Preview', 8: 'Scaled'}},
                                                   {'stream_cache_sync_state': {1: 'Synced'}}]},
                     'items': {'updates': [{'items_itemType': {32: 'Folder', 3: 'Image', 5: 'Video', 1: 'Document'}}, 
                                           {'items_deletedState': {1: 'Deleted', 0: 'Live'}},
                                           {'items_vaultType': {0: 'Not in Vault', 1: 'Vault Folder', 2: 'In Vault'}}]}}

        for col in ['views','item_moves','comment','vault','locations','drives']:
            table_dict[col] = dict()

        # generate dataframes and perform modifications
        for table_name, values in table_dict.items():
            table_dict[table_name]['df'] = pd.read_sql_query("SELECT * FROM {}".format(table_name), conn)

            df = table_dict[table_name]['df']
            # Make column names unique to prevent conflicting nomenclature
            for col in df.columns.values.tolist():
                df.columns = df.columns.str.replace(col, '{}_{}'.format(table_name, col))

            if not table_dict[table_name]['df'].empty:
                if 'updates' in table_dict[table_name]:
                    for update in table_dict[table_name]['updates']:
                        for column, vals in update.items():
                            for old_val, new_val in vals.items():
                                df.loc[df[column] == old_val, column] = new_val

        # gather some metadata
        table_dict['metadata'] = dict()
        for col in 'drives_accountId', 'drives_driveDisplayName', 'drives_drivePath', 'drives_driveType':
            table_dict['metadata'][col] = table_dict['drives']['df'].loc[0, col]

        # perform all required joins
        table_dict['merged'] = (pd.merge(table_dict['items']['df'], table_dict['item_moves']['df'],
                                left_on='items__id', right_on='item_moves_itemRId', how='outer'))

        table_dict['merged'] = (pd.merge(table_dict['merged'], table_dict['stream_cache']['df'],
                                left_on='items__id', right_on='stream_cache_parentId', how='outer'))

        table_dict['merged'] = (pd.merge(table_dict['merged'], table_dict['views']['df'],
                                left_on='stream_cache_parentId', right_on='views_itemId', how='outer'))

        table_dict['merged'] = (pd.merge(table_dict['merged'], table_dict['locations']['df'],
                                left_on='items_items_locationId', right_on='locations__id', how='outer'))

        table_dict['merged'] = (pd.merge(table_dict['merged'], table_dict['comment']['df'],
                                left_on='items__id', right_on='comment_itemId', how='outer'))

        # Get folder names as a dictionary with their associated integer so we can update out merged dataframe
        folders_dict = pd.Series(table_dict['merged'].items_items_resourceIdAlias.values, index=table_dict['merged'].items__id).to_dict()
        folders_dict = {int(k):v for k,v in folders_dict.items() if not pd.isna(k) and v is not None}
        # Now we can update the columns item_moves_sourceParentItemId and views_parentId
        for col in ['item_moves_sourceParentItemId', 'views_parentId']:
            table_dict['merged'][col].replace(folders_dict, inplace=True)

        table_dict['merged']['items_size_mb'] = round(pd.to_numeric(table_dict['merged']['items_size'], errors='coerce') / (1024*1024), 2)
        table_dict['merged']['items_size_gb'] = round(pd.to_numeric(table_dict['merged']['items_size'], errors='coerce') / (1024*1024*1024), 4)
        table_dict['merged']['cache_size_mb'] = round(pd.to_numeric(table_dict['merged']['stream_cache_progress'], errors='coerce') / (1024*1024), 2)

        cols_to_keep_dict = {'items__id': 'id',
                             'items_name': 'File Name',
                             'items_extension': 'Extension', 
                             'items_itemType': 'Type',
                             'items_mediaDuration': 'Duration',
                             'views_parentId': 'Parent Folder',
                             'items_items_resourceIdAlias': 'Directory',
                             'items_creationDate': 'Creation Date',
                             'items_dateTaken': 'Date Taken (EXIF)',
                             'items_itemDate': 'Date',
                             'items_deletedFromLocation': 'Deleted From',
                             'item_moves_sourceParentItemId': 'Moved from',
                             'items_deletedState': 'Deleted State',
                             'items_ownerName': 'Owner Name',
                             'items_size': 'Size (Bytes)',
                             'items_size_mb': 'Size (MB)',
                             'items_size_gb': 'Size (GB)',
                             'items_sha1Hash': 'SHA1',
                             'items_width': 'Width',
                             'items_height': 'Height',
                             'items_cameraModel': 'Camera Model',
                             'items_location': 'Location',
                             'items_altitude': 'Altitude',
                             'items_latitude': 'Latitude',
                             'items_longitude': 'Longitude',
                             'locations_city': 'City',
                             'locations_countryOrRegion': 'Country',
                             'locations_street': 'Street',
                             'items_favoriteRank': 'Favourite Rank',
                             'items_sharedByDisplayName': 'Shared by (Name)',
                             'items_sharedByEmail': 'Shared by (Email)',
                             'items_dateShared': 'Date Shared',
                             'items_vaultType': 'Vault Type',
                             'stream_cache_streamType': 'Cache Type',
                             'stream_cache_sync_state': 'Cache Sync State',
                             'stream_cache_progress': 'Cache Size (Bytes)',
                             'cache_size_mb': 'Cache Size (MB)',
                             'stream_cache_stream_location': 'CacheLocation',
                             'stream_cache_last_access_date': 'Cache Last Access Date',
                             'items_commentCount': 'Comment Count',
                             'comment_content': 'Comment',
                             'comment_createdDateTime': 'Comment Created Date',
                             'comment_creatorEmail': 'Comment Created By',
                             'items_ownerCid': 'CID'}

        df = table_dict['merged']
        meta = table_dict['metadata']

        # Tidy up the merged table
        df = df[list(cols_to_keep_dict.keys())]
        df.rename(cols_to_keep_dict, axis=1, inplace=True)

        # Convert timestamps
        for col_header in ['Creation Date', 'Date', 'Date Taken (EXIF)', 'Cache Last Access Date', 'Comment Created Date']:
            table_dict['merged'] = timestamp_converter(df, col_header, 'ms')

        # There are duplicate entries since stream cache can be saved in 3 different forms.
        # We can sort the merged table by the stream cache size (ascending) and keep the last,
        # largest streamcache file for the item in our report.
        df.sort_values('Cache Size (Bytes)', ascending=True, inplace=True)
        df.drop_duplicates(subset=['id'], keep='last', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # drop empty rows
        df.replace("", float("NaN"), inplace=True)
        df.dropna(subset=['id'], inplace=True)
        df.replace(np.nan, '', inplace=True)

        # Fetch some additional metadata
        meta['OneDrive Account Created'] = (df.loc[df['Directory'] == 'root', 'Creation Date'].iloc[0])
        meta['Comment Count'] = df['Comment'].replace(r'^\s*$', np.nan, regex=True).count()
        meta['Vault File Count'] = int(df['Vault Type'].str.count('In Vault').sum())

        # We meed tp remove some chars that affect our filepath links
        df['CacheLocation'].replace('%', '_', regex=True, inplace=True)

        # get the relative path of the cache files for our report
        rel_filepaths = dict()
        for file in files_dict['cache']:
            if file.endswith('.bin'):
                # We can chnage bin file names to jpeg
                os.rename(file, file.replace('.bin', '.jpg'))
            split_path = normpath(file).split(os.sep)
            idx = split_path.index('report_files')
            # we will use the key to find the file in our dataframe and then append the relative path of the cache
            rel_filepaths['/'.join(split_path[-3:])] = relpath('\\'.join(split_path[idx+1:])).replace('.bin', '.jpg')

        for k, v in rel_filepaths.items():
            df.loc[df['CacheLocation'].apply(lambda x: k in x), 'CacheLocation'] = v

        df.sort_values('Directory', ascending=False, inplace=True)
        return table_dict


class MakeReport:
    def __init__(self, *args):
        self.output_dir, self.df, self.metadata, self.has_media = args

        # Make CSV from dataframe
        self.df.to_csv(pj(self.output_dir, 'OneDrive.csv'))

        self.report_title = 'OneDrive'
        self.report_name = '{}.html'.format(self.report_title)

        copy_tree(report_components, self.output_dir)
        os.rename(pj(self.output_dir, 'index.html'), pj(self.output_dir, self.report_name))
        self.json_data = self.generate_json_object()
        self.apply_meta_data(["casename", "Report", self.report_title])

    def build(self):
        for md_key, md_val in self.metadata.items():
            self.apply_meta_data([md_key, md_key, md_val])

        if self.has_media:
            self.apply_column_data("Media Source", [{"name": "Miniature", "displayName": "Thumbnail",
                                                     "filter": 'null', "visibleIndex": 0},
                                                    {"name": "MediaLink", "displayName": "Link",
                                                     "filter": 'null', "visibleIndex": 1}])

        column_data = list()
        for col_count, col in enumerate(self.df.columns.values.tolist(), start=2):
            column_data.append({"name": col, "displayName": col, "filter": 'null', "visibleIndex": col_count})

        self.apply_column_data("File Information", column_data)

        row_data = []
        self.df.reset_index(inplace=True)
        for index, row in self.df.iterrows():
            row_ = dict()
            if self.has_media and row['CacheLocation']:
                row_ = {"Miniature": generate_thumbnail(pj(self.output_dir, row['CacheLocation'])),
                        "MediaLink": row['CacheLocation']}
            for k, v in row.items():
                if k not in row_:
                    try:
                        v = v.decode('utf8')
                    except:
                        pass
                    row_.update({k: v})
            row_data.append(row_)

        self.apply_row_data(row_data)
        self.convert_json_to_javascript(json.dumps(self.json_data, cls=NpEncoder), self.output_dir)
        wb.open(self.output_dir)

    def apply_meta_data(self, meta_list):
        self.json_data['window.caseData']['metaData'][meta_list[0]] = {"title": meta_list[1], "value": meta_list[2]}

    def apply_column_data(self, group_name, column_data):
        # Groups of columns can reside in the below column field.
        # The displayName is how the column will be named in the report.
        # The name is an object name.
        # The filter allows the user to filter the column.
        # visibleIndex is the column number (starts at 0)
        # filter is either 'null' or 'selection'
        # e.g. {"groupName": "Media Source", "columns": [{"name": "Miniature", "displayName": "Thumbnail",
        # "filter": 'null', "visibleIndex": 0}]}
        self.json_data['window.caseData']['columns'].append({"groupName": group_name, "columns": column_data})

    def apply_row_data(self, row_data):
        if isinstance(row_data, dict):
            self.json_data['window.caseData']['rows'].append(row_data)
        elif isinstance(row_data, list):
            for row in row_data:
                self.json_data['window.caseData']['rows'].append(row)

    def generate_json_object(self):
        json_data = dict()
        json_data['window.caseData'] = dict()
        json_data['window.caseData']['metaData'] = dict()
        json_data['window.caseData']['columns'] = list()
        json_data['window.caseData']['rows'] = list()
        json_data['window.caseData']['logo'] = {"width": 320, "height": 65,
                                                "image": (base64.b64encode(open(resource_path('ControlF_R_RGB.png'), 'rb').read()).decode())}
        # casename is a default key that must remain. It is used to display a 'title' for the report
        return json_data

    def convert_json_to_javascript(self, json_data, output_dir):
        newdata = (json_data.replace('{"window.caseData":', 'window.caseData =')
                   .replace('True', 'true')
                   .replace('False', 'false')
                   .replace('None', 'null')
                   .replace('True', 'true')
                   .replace('}}}', '},};'))

        data_file = pj(output_dir, 'data_{}.js'.format(self.report_name.split('.')[0]))
        f = open(data_file, 'w')
        f.write(newdata)
        f.close()

        with open(pj(output_dir, self.report_name), 'r', encoding='utf8') as index_file:
            content = index_file.read()
            content = content.replace('<script src="data.js"></script>',
                                      '<script src="{}"></script>'.format(basename(data_file)))

        with open(pj(output_dir, self.report_name), 'w', encoding='utf8') as index_file:
            index_file.write(content)


if __name__ == '__main__':
    print("\n\n"
          "                                                        ,%&&,\n"
          "                                                    *&&&&&&&&,\n"
          "                                                  /&&&&&&&&&&&&&\n"
          "                                               #&&&&&&&&&&&&&&&&&&\n"
          "                                           ,%&&&&&&&&&&&&&&&&&&&&&&&\n"
          "                                        ,%&&&&&&&&&&&&&&#  %&&&&&&&&&&,\n"
          "                                     *%&&&&&&&&&&&&&&%       %&&&&&&&&&%,\n"
          "                                   (%&&&&&&&&&&&&&&&&&&&#       %&%&&&&&&&%\n"
          "                               (&&&&&&&&&&&&&&&%&&&&&&&&&(       &&&&&&&&&&%\n"
          "              ,/#%&&&&&&&#(*#&&&&&&&&&&&&&&%,    #&&&&&&&&&(       &&&&&&&\n"
          "          (&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#          %&&&&&&&&&(       %/\n"
          "       (&&&&&&&&&&&&&&&&&&&&&&&&&&&&&(               %&&&&&&&&&/\n"
          "     /&&&&&&&&&&&&&&&&&&%&&&&&&&%&/                    %&&&&&,\n"
          "    #&&&&&&&&&&#          (&&&%*                         #,\n"
          "   #&&&&&&&&&%\n"
          "   &&&&&&&&&&\n"
          "  ,&&&&&&&&&&\n"
          "   %&&&&&&&&&                            {}\n"
          "   (&&&&&&&&&&,             /*           Version: {}\n"             
          "    (&&&&&&&&&&&/        *%&&&&&#\n"
          "      &&&&&&&&&&&&&&&&&&&&&&&&&&&&&%\n"
          "        &&&&&&&&&&&&&&&&&&&&&&&&&%\n"
          "          *%&&&&&&&&&&&&&&&&&&#,\n"
          "                *(######/,".format(__description__, __version__))
    print('\n\n')

    try:
        input_archive = sys.argv[1]
    except IndexError:
        print('Argument 1: must be the file path to the OneDrive archive')
        sys.exit()

    try:
        output_dir = sys.argv[2]
    except IndexError:
        print('Argument 2: must be the output directory (existing)')
        sys.exit()

    if isfile(input_archive) and (zipfile.is_zipfile(input_archive) or tarfile.is_tarfile(input_archive)):
        pass
    else:
        print('Argument 1: Not an archive (zip/tar)')
        sys.exit()

    if isdir(output_dir):
        pass
    else:
        print('Argument 2: The output location is not a directory or it does not exist')
        sys.exit()

    OneDriveParser(input_archive, output_dir)

