# openVINO のお試しプログラム類(openVINO 2022.1.0 対応版)

[ov_trial](https://github.com/ippei8jp/ov_trial) のopenVINO 2022.1.0 対応版です。  
インストール方法も変わったので、それまでのしがらみを捨てて  
以前のバージョン依存部分を削除してすっきりさせてみました。  

## ディレクトリ構成

| ディレクトリ        | 内容                                                                        |
|---------------------|-----------------------------------------------------------------------------|
| images              | テスト用画像保存用                                                          |    
| convert_model_ssd   | インターネット上で配布されているモデルファイルをIR形式に変換する(SSD)       |  
| ssd                 | SSDを実行する                                                               |
| convert_model_ssd   | インターネット上で配布されているモデルファイルをIR形式に変換する(顔認識)    |  
| face                | 顔認識を実行する                                                            |


## 事前準備
### openVINOのインストール  

とりあえず、pyenvで仮想環境作ってインストールしてみる。  
[Intel® Distribution of OpenVINO™ Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)
でDevtools、Linux、2022.1、PIP、Caffe/Tensorflow2.X/ONNXを選ぶと
「Download Intel® Distribution of OpenVINO™ Toolkit」の欄に必要なpipコマンドが表示されるので、それを実行すれば良い。  
今回はC/C++使わないので、Runtimeの方はインストールしなくても大丈夫。
(NCS2等を使いたい場合はドライバインストールに必要だけど)  

> 「Don’t miss out」の欄にメールアドレスを入力してSubmitをクリックすると色々なメールが送られてくるようになるけど、必須ではない。  
> (以前は登録しないとダウンロードさせてくれなかったけど、今は完全フリーになったらしい)    

以下ではonnx,tensorflow2,caffeを選択した場合。  
openVINO 2022.1.0はpython 3.10をサポートしてないので、現時点でのそれ以下の最新版3.9．13を使うことにする。  

```bash
pyenv install 3.9.13
pyenv virtualenv 3.9.13 openvino_2022.1
pyenv local openvino_2022.1 
pip install --upgrade pip setuptools wheel
pip install openvino-dev[onnx,tensorflow2,caffe]==2022.1.0
```
