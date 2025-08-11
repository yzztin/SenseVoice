## npu 调用

### 1. 示例环境

- npu 910b4
- 驱动 25.0.rc1.1
- CANN 8.0.RC2
- python: 3.11.13
- torch 2.3.1, torch-npu 2.3.1.post6
- torchaudio 2.3.0
- funasr 1.2.6


### 2. 修改点

- 完全基于 `torch-npu` 插件的支持，关键是需要在每次 `import torch` 之前，先行做到 `import torch_npu`
- 以 `demo1.py` 为例，原始代码中开头调用了 `from funasr import AutoModel`（执行了 `import torch`），所以需要在其之前先导入：
    ```
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    ```
- 同理其他调用代码也是如此，顺利的话，只需要写上上面两行代码就可以正常使用 npu 推理计算了，随后可通过 `npu-smi info` 查看进程的显存占用


### 3. 使用

1. 安装依赖：`requirements.txt` 是必装依赖，`reuqiremens_for_npu.txt` 中是昇腾环境中可能需要的依赖，按需安装即可
2. 执行测试： `python demo1.py`, `python demo2.py`, `python webui.py`
3. 启动服务化接口：按需修改 `run_api.sh` 中的变量，执行 `chmod +x run_api.sh && ./run_api.sh`
4. 接口测试：
    ```
    curl --location --request POST 'http://127.0.0.1:12345/api/v1/asr' \
    --form 'files=@"./test_files/korean1.wav"' \
    --form 'files=@"./test_files/korean2.mp3"' \
    --form 'lang="ko"'
    ```

### 参考链接：
- https://bbs.huaweicloud.com/blogs/439085
- https://github.com/FunAudioLLM/SenseVoice/pull/26
