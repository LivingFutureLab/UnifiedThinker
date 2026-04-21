from torch.utils.data import DataLoader


def load_data(args):
    dataset = load_dataset(**args.train_data)
    dataloader = load_dataloader(dataset, **args.dataloader)
    return dataset, dataloader

def load_data_und(args):
    # for understand data
    dataset = load_dataset(**args.train_data_und)
    dataloader = load_dataloader(dataset, **args.dataloader)
    return dataset, dataloader

def load_data_gen(args):
    # for understand data
    dataset = load_dataset(**args.train_data_gen)
    dataloader = load_dataloader(dataset, **args.dataloader)
    return dataset, dataloader

def load_valid_data_gen(args):
    # for understand data
    dataset = load_dataset(**args.valid_data_gen)
    dataloader = load_dataloader(dataset, **args.valid_dataloader)
    return dataset, dataloader

def load_dataset(data_type, data_params):    
    # if data_type == 'local_t2i':
    #     from src.data.local_t2i_data import LocalT2IDataset # for notebook debug
    #     dataset = LocalT2IDataset(**data_params)
    if data_type == "local_t2i_edit":
        from src.data.local_t2i_edit_data import LocalT2iEditDataset
        dataset = LocalT2iEditDataset(**data_params)
    elif data_type == "local_und":
        from src.data.local_und_data import LocalUndDataset
        dataset = LocalUndDataset(**data_params)
    elif data_type == "local_thinker":
        from src.data.local_thinker_data import LocalThinkerDataset
        dataset = LocalThinkerDataset(**data_params)
    
    elif data_type == "odps_und":
        from src.data.odps_und_data import UndDistributeDataset
        dataset = UndDistributeDataset(**data_params)
    # elif data_type == "odps_gen":
    #     from src.data.odps_gen_data import GenDistributeDataset
    #     dataset = GenDistributeDataset(**data_params)
    elif data_type == "odps_t2i_edit":
        from src.data.odps_t2i_edit_data import T2iEditDistributeDataset
        dataset = T2iEditDistributeDataset(**data_params)
    else:
        raise NotImplementedError(f"Data type {data_type} is not supported")
        
    return dataset


def load_dataloader(dataset, **kwargs):
    # ailake
    if getattr(dataset, "new_epoch", None):
        dataset.new_epoch()

    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        **kwargs,
    )
    return dataloader
