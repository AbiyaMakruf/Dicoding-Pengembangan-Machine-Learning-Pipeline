# Penilaian Proyek
Proyek ini berhasil mendapatkan bintang 3/5 pada submission dicoding course Machine Learning Operations (MLOps).
![Penilaian Submission](https://raw.githubusercontent.com/AbiyaMakruf/Dicoding-PengembanganMachineLearningPipeline/main/image/nilai.png)

Kriteria agar mendapatkan bintang tambahan yang tidak saya kerjakan adalah: 
- Memanfaatkan komponen Tuner untuk menjalankan proses hyperparameter tuning secara otomatis.
- Menambahkan sebuah berkas notebook untuk menguji dan melakukan prediction request ke model serving yang telah dibuat.

# Laporan Proyek
Nama: Muhammad Abiya Makruf

Username dicoding: abiyamf

| | Deskripsi |
| ----------- | ----------- |
| Dataset |Dataset yang digunakan adalah [Weather Type Classification](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification/data) Dataset ini berisi berbagai variabel cuaca yang digunakan untuk mengklasifikasikan jenis cuaca.|
| Masalah | Masalah yang diangkat dalam proyek ini adalah klasifikasi jenis cuaca berdasarkan beberapa fitur cuaca. Mengingat kondisi cuaca sangat penting untuk berbagai keperluan seperti pertanian, penerbangan, dan perencanaan aktivitas sehari-hari, memiliki model yang dapat mengklasifikasikan jenis cuaca berdasarkan data cuaca historis sangatlah bermanfaat. |
| Solusi machine learning | Solusi machine learning yang diusulkan adalah membangun pipeline machine learning yang mencakup preprocessing data, pelatihan model, evaluasi model, dan deployment model menggunakan TensorFlow Serving. Pipeline ini akan menggunakan TFX (TensorFlow Extended) untuk mengelola alur kerja. |
| Metode pengolahan | Preprocessing Data <br> Metode pengolahan data yang digunakan meliputi: <br> 1. Normalisasi: Fitur numerik dinormalisasi menggunakan Z-score normalization. <br> 2. Encoding: Fitur kategorikal diencoding menggunakan metode one-hot encoding. |
| Arsitektur model | Arsitektur model yang digunakan adalah model neural network dengan beberapa lapisan fully connected. Berikut adalah detail arsitektur model: <br> 1. Input Layer: Mengambil fitur-fitur cuaca yang telah ditransformasikan. <br> 2. Hidden Layer: Dua lapisan dense menggunakan aktivasi ReLU. <br> 3. Output Layer: Lapisan dense dengan 4 unit dan fungsi aktivasi softmax untuk klasifikasi jenis cuaca. <br> 4. Optimizer: Adam. <br> 5. Loss: categorical_crossentropy. <br> 6. Metrics: CategoricalAccuracy. |
| Metrik evaluasi | Metrik yang digunakan untuk mengevaluasi performa model adalah: <br> 1. Categorical Accuracy: Akurasi khusus untuk masalah klasifikasi dengan lebih dari dua kelas. <br> 2. Confusion Matrix: Matriks yang menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas. |
| Performa model | Performa model dievaluasi menggunakan dataset evaluasi. Hasil evaluasi menunjukkan bahwa model memiliki akurasi yang tinggi dalam mengklasifikasikan jenis cuaca dengan menggunakan fitur-fitur cuaca yang tersedia. Hasil hyperparameter tuning menunjukkan bahwa model dengan arsitektur tertentu memberikan performa terbaik dengan akurasi mencapai lebih dari 90%. |

# How to reproduce
## Membuat Virtual Environtment

- Buat virtual environtment dengan menjalankan perintah berikut 

    ```
    conda create --name mlops-tfx python==3.9.15
    ```

- Selanjutnya aktifkan environtment 

    ```
    conda activate mlops-tfx
    ```

## Mempersiapkan requirements
- Clone repository ini dengan menjalankan 

    ```
    https://github.com/AbiyaMakruf/Dicoding-PengembanganMachineLearningPipeline.git
    ```

- Pastikan sudah pada environtment mlops-tfx kemudian jalankan 

    ```
    pip install -r requirements.txt
    ```

## Menjalankan docker
- Jalankan docker dekstop

- Buat docker image dengan menjalankan
    ```
    docker build -t weather-model .
    ```

- Jalankan docker image dengan menjalankan
    ```
    docker run -p 8080:8501 weather-model
    ```

- Apabila ingin memastikan versi model yang digunakan oleh TF-Sering, jalankan 
    ```python
    import requests
    from pprint import PrettyPrinter
    
    pp = PrettyPrinter()
    pp.pprint(requests.get("http://localhost:8080/v1/models/weather_classification_model").json())
    ```

- Apabila berjalan dengan lancar maka akan menghasilkan keluaran sebagai berikut
    ```
    {'model_version_status': [{'state': 'AVAILABLE',
                            'status': {'error_code': 'OK', 'error_message': ''},
                            'version': '1722275990'}]}
    ```
