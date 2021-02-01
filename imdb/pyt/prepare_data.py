import os

from imdb.data import upload_to_s3, IMDBDataSerializer
from imdb.pyt.code.config import S3_DATA_BUCKET, DATA_PATH, CACHE_PATH

s = IMDBDataSerializer(data_dir=os.path.join('imdb', 'data'), out_dir=DATA_PATH)
s.load().save()

for dr in [DATA_PATH, CACHE_PATH]:
    for filename in os.listdir(dr):
        f = os.path.join(dr, filename)
        print(f'Uploading {f}')
        upload_to_s3(f, S3_DATA_BUCKET)
