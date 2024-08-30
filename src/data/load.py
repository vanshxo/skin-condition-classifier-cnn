import gdown
import pathlib
from zipfile import ZipFile

path='https://drive.google.com/uc?id=1MS_bEkYsLCp0M4gf6fN_ZhK4zhaO9jOs'

def download_data(output_path):
    gdown.download(path,output=output_path,quiet=False)

def main():
    curr_dir=pathlib.Path().cwd().as_posix()
    output_path=curr_dir+'/'+'data/raw/data.zip'
    extract_to_path=curr_dir+'/'+'data/raw/'
    download_data(output_path=output_path)
    ZipFile(output_path).extractall(extract_to_path)


if __name__ =='__main__':
    main()
