import os
import time
from azure.storage.blob import BlockBlobService, PublicAccess

player_name_train_dir = r'E:\Data\Overwatch\training_data\player_status'

def upload_player_status():

    block_blob_service = BlockBlobService(account_name='omnicintellige6390360867', account_key='Z1u6QIYTp8QOP+ECSUcyCtNk+zBLBoq7jdsyvG64fbVsv/0MBzlHxFKNObVB3MVn9MZE2rlVQJXSv5TaBw4dHQ==')
    # Create a container called 'quickstartblobs'.
    container_name ='playerstatus'
    block_blob_service.create_container(container_name)

    # List the blobs in the container
    print("\nList blobs in the container")
    generator = block_blob_service.list_blobs(container_name)
    uploaded = set()
    for blob in generator:
        uploaded.add(blob.name)
        print("\t Blob name: " + blob.name)

    for f in os.listdir(player_name_train_dir):
        #if f.endswith('.hdf5') and int(os.path.splitext(f)[0]) > 10000:
        #    continue
        if f in uploaded:
            continue
        print('Uploading '+ f)
        begin = time.time()
        block_blob_service.create_blob_from_path(container_name, f, os.path.join(player_name_train_dir, f))
        print('Took {} seconds'.format(time.time()-begin))

if __name__ == '__main__':
    upload_player_status()