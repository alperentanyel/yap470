# YAP470 Topic Classifier Project - Jupyter Notebook Descriptions

Bu proje, metin sınıflandırma için hibrit özellik vektörizasyonu kullanan makine öğrenmesi modellerini içermektedir.

## Notebook Dosyaları

### 1. topic_classifier_training.ipynb
**Amaç:** Model eğitimi ve hibrit feature vektörizasyonu pipeline'ı

**Ana Özellikler:**
- Hibrit Vektör Oluşturma: GloVe embeddings + TF-IDF ağırlıklı kategori vektörleri (100D → 1400D)
- Eksik Kelime İşleme: GloVe'de olmayan kelimeler için kategorsel TF-IDF bazlı KNN ile vektör oluşturma
- 4 farklı ML modeli: LogisticRegression, SVM, MLP, GradientBoosting
- PCA boyut indirgeme: 1400D → 100D (hız optimizasyonu için)
- Hızlı mod: Önceden optimize edilmiş hiperparametreler (random search atıldı)
- Otomatik GloVe indirme ve yükleme

**Workflow:**
1. Veri yükleme (train.csv, test.csv)
2. GloVe embeddings yükleme (100D)
3. Kategori-bazlı TF-IDF skorları hesaplama
4. Hibrit feature vektörleri oluşturma
5. LogisticRegression eğitimi (tam boyutlu vektörler)
6. PCA boyut indirgeme uygulaması
7. Diğer modellerin eğitimi (PCA vektörleri)
8. Performans analizi ve model kaydetme

**Çıktılar:**
- Eğitilmiş modeller (models/ klasöründe)
- Feature vectorizer ve PCA transformer
- Training konfigürasyonu
- TF-IDF skorları ve GloVe embeddings

**Kategoriler:** 14 farklı kategori (0-13 arası indeksler)

### 2. topic_classifier_test.ipynb
**Amaç:** Eğitilmiş modelleri test etme ve karşılaştırma

**Ana Özellikler:**
- Model Yükleme: Tüm eğitilmiş modelleri otomatik yükleme
- Dosyadan Test: CSV dosyasından test verilerini yükleme
- Manuel Test: Kullanıcı tarafından yazılan metinleri test etme
- Model Karşılaştırması: Tüm modellerin performansını karşılaştırma
- Detaylı Analiz: Confusion matrix, classification report
- Hata Analizi: Kategori bazlı doğruluk ve hata paternleri
- Hız Testi: Model tahmin hızlarını ölçme

**Test Tipleri:**
1. Hızlı test (örnek metinler)
2. Dosyadan test (test.csv)
3. Manuel metin girişi
4. Kapsamlı model karşılaştırması

**Çıktılar:**
- Model doğruluk skorları
- Kategori bazlı performans analizi
- Confusion matrix tabloları
- Hata analiz raporları
- Model hız karşılaştırması

## Kullanım Sırası

1. **İlk olarak topic_classifier_training.ipynb çalıştırın:**
   - Modelleri eğitir ve kaydeder
   - Gerekli bileşenleri oluşturur
   - Uzun sürer (yaklaşık 5-6 saat)

2. **Sonra topic_classifier_test.ipynb çalıştırın:**
   - Eğitilmiş modelleri test eder
   - Farklı test senaryolarını dener
   - Performans analizlerini görüntüler

## Gerekli Dosyalar

**Veri Dosyaları:**
- archive/train.csv (eğitim verisi) (DBPedia Ontology)
- archive/test.csv (test verisi)

**Model Dosyaları (training sonrası oluşur):**
- models/feature_vectorizer.pkl (metin → vektör dönüşümü)
- models/pca_transformer.pkl (boyut indirgeme)
- models/training_config.pkl (konfigürasyon)
- models/[model_name]_model.pkl (eğitilmiş modeller)
- models/glove_embeddings.pkl (word embeddings)
- models/tfidf_word_scores.pkl (TF-IDF skorları)

**GloVe Dosyaları (otomatik indirilir):**
- glove/glove.6B.100D.txt (word embeddings)

## Teknik Detaylar

**Hibrit Vektör Yapısı:**
- GloVe 100D base vector
- 14 kategori × 100D = 1400D hibrit vektör
- Her kategori için TF-IDF ağırlıklı vektör bileşeni
- PCA ile 100D'a indirgeme (hız için)

**Model Mimarisi:**
- LogisticRegression: Tam boyutlu hibrit vektörler (1400D)
- SVM/MLP/GradientBoosting: PCA indirgenmiş vektörler (100D)

**Performans Optimizasyonları:**
- Önceden tanımlı hiperparametreler (hızlı mod)
- TF-IDF cache sistemi
- Progress bar'lar
