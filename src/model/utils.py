import os
import torch
import time
import shutil
import oss2
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def upload_file_oss2cpfs(rel_path, cpfs_path, oss_path, bucket):
    try:
        cpfs_file = os.path.join(cpfs_path, rel_path)
        oss_file = os.path.join(oss_path, rel_path)
        os.makedirs(os.path.dirname(cpfs_file), exist_ok=True)
        if not os.path.exists(cpfs_file):
            print("\n{} -> {}".format(oss_file, cpfs_file))        
            bucket.get_object_to_file(oss_file, cpfs_file)
        else:
            print(f"{cpfs_file} exist.")
    except Exception as e:
        print(f"Error of {str(e)}")
        
def download_model_weight_oss(oss_path):
    assert oss_path.startswith("oss://tstar-image-dataset/"), "wrong of model_path: {}".format(oss_path)
    oss_path = oss_path.replace("oss://tstar-image-dataset/", "")
    cpfs_path = os.path.join("/data/xpfs_0/", oss_path)
    if os.path.exists(cpfs_path):
        return cpfs_path
    
    cpfs_path = os.path.join("./ckpt_temp", oss_path)
    
    global_rank = int(os.getenv("LOCAL_PROCESS_RANK", "0"))
    if global_rank == 0:        
        rel_files = []
        # for obj in bucket.list_objects(prefix=oss_path).object_list:  # 会限制文件数目 100 个
        #     # obj is an ObjectInfo instance. We are interested in its key.
        #     file_name = obj.key 
        #     if "." in os.path.basename(file_name):
        #         # is a file
        #         relative_path = os.path.relpath(file_name, oss_path)
        #         rel_files.append(relative_path)  
        for obj in oss2.ObjectIterator(bucket, prefix=oss_path):
            file_name = obj.key 
            if "." in os.path.basename(file_name):
                # is a file
                relative_path = os.path.relpath(file_name, oss_path)
                rel_files.append(relative_path)  
            
        # 使用多线程下载文件
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(partial(upload_file_oss2cpfs, cpfs_path=cpfs_path, oss_path=oss_path, bucket=bucket), rel_files)
            
    return cpfs_path


def upload_file_cpfs2oss(rel_path, cpfs_path, oss_path, bucket):
    try:
        cpfs_file = os.path.join(cpfs_path, rel_path)
        oss_file = os.path.join(oss_path, rel_path)
        print("\n{} -> {}".format(cpfs_file, oss_file))
        bucket.put_object_from_file(oss_file, cpfs_file)
    except Exception as e:
        print(f"Error of {str(e)}")


def upload_model_weight_oss(cpfs_path, oss_path):
    end_point = ''
    auth = ''
    bucket = oss2.Bucket(auth, end_point, 'tstar-image-dataset')
    
    assert oss_path.startswith("oss://tstar-image-dataset/"), "wrong of model_path: {}".format(oss_path)
    oss_path = oss_path.replace("oss://tstar-image-dataset/", "")
        
    rel_files = []
    for root, dirs, files in os.walk(cpfs_path):
        for filename in files:
            # 构造文件的完整绝对路径
            full_path = os.path.join(root, filename)
            # 计算相对于起始目录的路径
            relative_path = os.path.relpath(full_path, cpfs_path)
            rel_files.append(relative_path)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(partial(upload_file_cpfs2oss, cpfs_path=cpfs_path, oss_path=oss_path, bucket=bucket), rel_files)
    

def download_model_weight(model_path):
    """
    download model ckpt to local file.
        now support: mos
        TODO: support oss link
    """
    xpfs_path = os.path.join("/data/xpfs_0/", model_path)
    if os.path.exists(xpfs_path):
        print("model exists on xpfs:", xpfs_path)
        model_path = xpfs_path
    else:
        root_dir = f"./ckpt_temp_{model_path.split('/')[-1]}"
        label_path = os.path.join(root_dir, "success")
        global_rank = int(os.getenv("LOCAL_PROCESS_RANK", "0"))
        if global_rank == 0:
            if model_path.startswith("model."):
                os.environ["USER_ID"] = "147878"
                # 以前的下载方式，废弃！
                # from mdl.model_hub import ModelHubClient
                # model_hub = ModelHubClient()
                # model_hub.load_mos_model_to_dir(
                #     local_directory=root_dir, mos_model_uri=model_path
                # )
                from openlm_hub import repo_download

                print(f"Mos model loading to {root_dir}, From {model_path}")
                repo_download(repo_id=model_path, local_dir=root_dir)
            else:
                shutil.copytree(
                    "./data/" + model_path, root_dir, dirs_exist_ok=True
                )
                # shutil.copytree(model_path, root_dir, dirs_exist_ok=True)
            open(label_path, "w").close()
        else:
            while True:
                if os.path.exists(label_path):
                    break
                time.sleep(1.0)
        # torch.distributed.barrier()
        model_path = root_dir

    for root, dirs, files in os.walk(model_path):
        for file in files:
            print("files: ", os.path.join(root, file))

    return model_path
