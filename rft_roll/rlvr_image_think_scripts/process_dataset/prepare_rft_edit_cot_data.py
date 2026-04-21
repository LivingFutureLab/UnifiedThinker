#coding=utf-8
#jianchong.zq: 从 odps 表中拉 reason edit 相关数据, 过滤掉没有cot的data, 数据格式对齐 roll。

import os
import common_io
import json
from odps import ODPS
    
def reason_edit_data_from_table(table_name="tbstar_image_generation_original_database_new", space="future_xingchen_dev"):
    
    def read_one_partion(partition):
        if partition is None:
            reader = common_io.table.TableReader(
                f"odps://{space}/tables/{table_name}",
                selected_cols="ref_imgs_oss_path,tgt_imgs_oss_path,text_inst,text_dec")
        else:
            reader = common_io.table.TableReader(
                f"odps://{space}/tables/{table_name}/{partition}",
                selected_cols="ref_imgs_oss_path,tgt_imgs_oss_path,text_inst,text_dec")

        total_records_num = reader.get_row_count()
        print("total_records_num: {} from partition: {}".format(total_records_num, partition))
        
        datas = []    
        while True:
            try:
                records = reader.read(1)[0]
                datas.append({
                    "ref_imgs_oss_path": records[0],
                    "tgt_imgs_oss_path": records[1],
                    "text_inst": records[2],
                    "text_dec": records[3]
                })
                
                if len(datas) % 1000 == 0:
                    print(f"{len(datas)} datas")
                    
            except common_io.exception.OutOfRangeException:
                break
            else:
                continue
        reader.close()
        return datas
        
    reason_edit_partitions = [
        "ds=20251103/part=reason_edit_itag_1980653650223742976-gemini_jc",
        "ds=20251103/part=reason_edit_itag_1980147097150980096-gemini_jc",
        "ds=20251103/part=reason_edit_itag_1978419717631283200-gemini_jc",
        "ds=20251103/part=reason_edit_itag_1983064215830040576-gemini_jc",
    ]
    datas = []
    for partition in reason_edit_partitions:
        datas += read_one_partion(partition)
        
        
    ## rlvr data format
    output_datas = []   # dict_keys(['data_source', 'images', 'prompt', 'ability', 'reward_model', 'extra_info'])
    for d in datas:
        ref_imgs_oss_path = json.loads(d["ref_imgs_oss_path"])
        tgt_imgs_oss_path = json.loads(d["tgt_imgs_oss_path"])
        text_inst = json.loads(d['text_inst'])
        if "cn_short" in text_inst:
            edit_prompt = text_inst["cn_short"]
        elif "en_short" in text_inst:
            edit_prompt = text_inst["en_short"]
        else:
            continue
        
        if "cn_cot" in text_inst:
            edit_prompt_cot = text_inst["cn_cot"]
        elif "en_cot" in text_inst:
            edit_prompt_cot = text_inst["en_cot"]
        else:
            continue
        
        output_datas.append({
            "ref_imgs": ref_imgs_oss_path,
            "tgt_images": tgt_imgs_oss_path,
            "edit_prompt": edit_prompt,
            "edit_prompt_cot": edit_prompt_cot,
            "task_type": "reason_edit",
            "domain": "reason_edit"
        })
        
    outfile = f"rft_edit_cot__task_reason__{len(output_datas) // 1000}k.jsonl"
    with open(outfile, 'w', encoding='utf-8') as f:
        for d in output_datas:
            f.write("{}\n".format(json.dumps(d, ensure_ascii=False)))
    print(f"write {len(output_datas)} samples to {outfile}")
    
if __name__ == '__main__':
    import pdb; pdb.set_trace()
    
    reason_edit_data_from_table()
    print()