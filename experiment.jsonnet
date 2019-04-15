{
    "train_data_path": "names/*.txt",
    
    "dataset_reader": {
        "type": "names-reader"
    },
    "model": {
        "type": "names-classifier",
        "word_embeddings": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 6
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 6,
            "hidden_size": 6
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 2,
        "sorting_keys": [["sentence", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": 10,
        "optimizer": {
            "type": "sgd",
            "lr": 0.1
        },
        "patience": 10
    }
}
