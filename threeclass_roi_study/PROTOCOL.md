# Frozen Predicted-ROI Üç Sınıflı Sınıflandırma Protokolü

Sürüm: 1.0.0  
Çalışma kimliği: `strict_predicted_roi_threeclass_4model_v1_0_0`  
Durum: Önceden tanımlı, iç veri üzerinde exploratory/post-hoc geliştirme ve internal testing  
Primer birim: Hasta  
Raporlama çerçevesi: CLAIM 2024 Update ve TRIPOD+AI 2024

## 1. Zorunlu önceki test kullanımı açıklaması

Normal ile papilödem+pseudopapilödemi ayıran kaynak ikili çalışmada aynı iç
test hasta üyelikleri daha önce açılmış, sonuçlar incelenmiş ve özgün üç tanı
sınıfına göre tanımlayıcı sonuçlar üretilmiştir. Bu nedenle bu üç sınıflı uzantı:

- bağımsız internal testing değildir;
- external testing değildir;
- doğrulayıcı/confirmatory çalışma değildir;
- **exploratory ve post-hoc iç çalışma** olarak raporlanacaktır.

Makale, özet, tablo, şekil ve makine-okunur metadata şu açıklamanın anlamını
korumak zorundadır:

> The same internal patient holdouts were previously opened and analysed for
> the binary normal-versus-disease study; this three-class extension is
> therefore exploratory, post-hoc, and not an independent internal or external
> validation.

Doğrulayıcı iddia ancak bu protokolün seçilmiş sistemi dondurulduktan sonra daha
önce hiç görülmemiş temporal veya dış merkez hasta kohortunda tek seferlik
değerlendirmeyle desteklenebilir.

## 2. Araştırma sorusu ve estimandlar

Primer soru, donmuş strict predicted-ROI lokalizasyon zinciri kullanıldığında
normal, papilödem ve pseudopapilödemin hasta düzeyinde ne kadar doğru
ayrılabildiğidir.

Primer estimand:

- birim: hasta;
- hedef popülasyon: abstention dahil tüm amaçlanan held-out test hastaları;
- müdahale/test: segmenter + ona ait yeni üç-logit `model_specific`
  strict-ROI classifier;
- endpoint: üç sınıflı failure-aware balanced accuracy;
- yorum: segmenter ve model-specific classifier'ın oluşturduğu tamamlanmış
  sistem etkisi.

Önceden tanımlı sekonder standartlaştırılmış estimand, her segmenterin ROI'lerini
ayrı eğitilmiş ortak ResNet-18 üç-logit classifier ile değerlendirir. Bu kol,
classifier mimarisini sabit tutarak predicted-ROI/segmenter etkisini
standartlaştırır. İki estimandın sıralamaları birbirine karıştırılmaz.

## 3. Veri, reference standard ve sınıflar

Kilitli manifest:

- 91 hasta;
- 182 göz;
- 1.274 frame;
- her hastada sağ ve sol göz;
- her gözde yedi benzersiz frame.

Sınıflar ve hasta sayıları:

| Kod | Sınıf | Hasta |
|---:|---|---:|
| 0 | normal | 48 |
| 1 | papilledema | 21 |
| 2 | pseudopapilledema | 22 |

CLAIM 2024 terminolojisine uygun olarak “ground truth” yerine “reference
standard” kullanılır. Tanı etiketi kilitli hasta düzeyi klinik tanıdır. ROI
reference standard'ı, kaynak segmenter geliştirme ve tahminler dondurulduktan
sonraki retrospektif lokalizasyon ölçümü içindir. Testte ROI seçme, düzeltme veya
fallback amacıyla kullanılamaz.

Yaş, cinsiyet ve diğer demografik değişkenler kilitli manifestte yoktur. Bu
durum eksik metadata ve fairness değerlendirmesi sınırlılığı olarak açıkça
raporlanır; yok değerler “normal dağılım” veya benzeri varsayımlarla
doldurulmaz.

## 4. Hasta-gruplu splitler

Split birimi hastadır. Bir hastanın iki gözü ve tüm frameleri aynı partition'da
kalır. Seed'ler tam olarak `17, 42, 2026, 3407, 9103`'tür. Kaynak ikili
çalışmanın SHA-256 kilitli üyelikleri byte-for-byte yeniden kullanılır; yeniden
örnekleme yapılmaz.

Her seed için train/validation/test hasta kotaları:

| Sınıf | Train | Validation | Test |
|---|---:|---:|---:|
| normal | 28 | 10 | 10 |
| papilledema | 13 | 4 | 4 |
| pseudopapilledema | 14 | 4 | 4 |

Aynı hastanın partition'lar arasında bulunmaması, iki gözün birlikte kalması,
etiket tutarlılığı, yedi benzersiz frame ve exact/perceptual near-duplicate
kontrolleri zorunludur. Beş test üyeliği kısmen örtüşür; beş bağımsız kohort gibi
havuzlanamaz.

## 5. Donmuş upstream yeniden kullanım kararı

Segmentasyon görevi üç tanı sınıfında aynı anatomik ROI'yi bulmaktır. Kaynak
segmenterler:

- tanı sınıflandırma kaybı olmadan yalnız segmentasyon supervision'ı ile
  eğitilmiştir;
- tanı sınıfına göre hasta düzeyinde tabakalı split kullanmıştır;
- classifier-train ROI'lerini beş inner fold ile tamamen hasta-out-of-fold
  üretmiştir;
- validation ROI'lerini outer-train full-fit segmenter ile üretmiştir;
- threshold, area ve component kurallarını testten önce kilitlemiştir.

Bu nedenle segmenter checkpoint'leri, segmenter lock'ları, train OOF ROI
indeksi ve validation ROI indeksi yeniden eğitilmeden kullanılabilir. Bunlar yeni
çalışmaya kopyalanmaz veya sessizce referans verilmez; her model×seed için
path, boyut ve SHA-256 içeren `import-upstream-development` receipt'i gerekir.

### 5.1 Geliştirme aşamasında izin verilen artifact'lar

1. Kaynak `build-rois` receipt'i
2. Segmenter lock'u
3. Seçilmiş segmenter checkpoint'i
4. Train OOF ROI index'i
5. Validation ROI index'i

### 5.2 Geliştirme aşamasında yasak artifact'lar

- İkili classifier checkpoint'leri veya classifier head'leri
- İkili logit ve olasılıklar
- İkili karar eşikleri
- İkili temperature/calibration değerleri
- Test ROI index'i
- Test evaluate receipt'i

Test ROI index'i ile kaynak evaluate receipt'i, yeni üç-sınıf validation
lock'larının tamamı doğrulanıp global test sentinel'i yazılana kadar
**okunamaz ve hash'lenemez**.

### 5.3 Upstream'in yeniden çalıştırılmasını gerektiren durumlar

Split, preprocessing, reference-standard ROI, component/area/threshold
kuralları, segmenter mimarisi veya classifier girdisi değiştirilirse upstream
artifact'lar artık aynı estimandı temsil etmez. Bu durumda yeni çalışma kimliği
altında segmenter ve inner cross-fitting yeniden çalıştırılmalıdır. Aynı şekilde
train classifier ROI'lerinin gerçek out-of-fold olmadığı anlaşılırsa import
fail-closed durur.

## 6. Üç sınıflı classifier geliştirme

İki core strateji vardır:

1. `model_specific`: YOLO26s, ViT-Method2, PVT-v2-B0 veya Hiera-tiny ailesine
   uygun yeni üç-logit head;
2. `standardized_resnet18`: her model×seed ROI akışı için ayrı eğitilen ortak
   ResNet-18 üç-logit classifier.

Her iki strateji aynı model×seed içinde byte-identical ROI tensoru, hard maske,
geçerlilik durumu ve abstention nedenini kullanır. İki-logit head üç logite
dönüştürülür; ikili head, eşik veya calibrator kullanılamaz.

Primer formülasyon doğrudan flat üç sınıflı softmax'tır. Hard
normal/anormal→papilödem/pseudopapilödem kaskadı primer değildir. Tanısal hata
kaynağını açıklamak için flat olasılıklardan:

`p(abnormal) = p(papilledema) + p(pseudopapilledema)`

ve

`p(papilledema | abnormal) = p(papilledema) /
[p(papilledema) + p(pseudopapilledema)]`

sekonder skorları hesaplanabilir.

Optimizasyon AdamW ve cosine annealing kullanır. Maksimum 60 epoch, patience 10
ve class-balanced cross-entropy kilitlidir. Ağırlıklandırma frame sayısına göre
değil, her tanı sınıfındaki her hastanın toplam katkısı eşit olacak biçimde
yapılır. Çok geçerli frame'i bulunan hasta optimizasyonu orantısız biçimde
domine edemez.

Checkpoint seçimi validation hasta macro-NLL ile yapılır. Eşitlik bozucu
validation hasta failure-aware balanced accuracy'dir. Test hiçbir eğitim,
early-stopping, architecture, threshold veya checkpoint kararında kullanılamaz.

## 7. Frame, göz ve hasta agregasyonu

Her geçerli frame için classifier üç elemanlı olasılık vektörü üretir:

`p_frame = [p_normal, p_papilledema, p_pseudopapilledema]`.

Bir gözde `K ≥ 4` geçerli frame varsa:

`p_eye = (1/K) × Σ p_frame`.

`K < 4` ise göz `ABSTAIN_INSUFFICIENT_FRAMES` olur.

İki göz de evaluable ise:

`p_patient = (p_right + p_left) / 2`.

Tek göz bile evaluable değilse hasta `ABSTAIN_INSUFFICIENT_EYES` olur. Tek-göz,
öteki classifier veya full-image fallback yoktur. Primer karar, hasta düzeyi
kalibre vektörde `argmax` ile verilir. Göz sonuçları zorunlu sekonderdir.

## 8. Kalibrasyon

Ham olasılıklar daima saklanır ve raporlanır. Tek parametreli scalar temperature
scaling, her model×seed×classifier strategy×raporlama seviyesi için yalnız
validation'da ve agregasyondan sonra fit edilir. Seviyeler göz ve hastadır.

Küçük validation örneklemi nedeniyle vector scaling, classwise temperature ve
Dirichlet calibration core analizde yasaktır. Herhangi bir tanı sınıfı
evaluable değilse veya optimizasyon tanımsızsa calibration `unavailable` olur;
`T=1` gibi örtük varsayılan yazılamaz ve ilgili validation lock yazılmadan çalışma
global test kapısında fail-closed durur. Böylece calibration arızası yeni bir
klinik abstention nedeni gibi gösterilemez. Testte calibration refit edilmez.

## 9. Abstention ve selective prediction

Primer politika yalnız yapısal strict-ROI kapılarını kullanır:

- gözde 4/7'den az geçerli ROI;
- hastada iki gözden en az birinin evaluable olmaması.

Bu abstention'lar primer failure-aware sonuçlarda yanlış karar sayılır.
Classifier belirsizliğine dayalı ek abstention yalnız önceden tanımlı sekonder
selective-prediction analizidir. Skor maksimum kalibre hasta olasılığıdır; varsa
operating threshold yalnız validation'da kilitlenir. Testte threshold
değiştirilemez.

Risk–coverage eğrisi bütün amaçlanan hastaları kapsar. Yapısal abstention'lar en
düşük confidence'a eklenir ve hata sayılır. Conditional doğruluk tek başına
primer performans olarak sunulamaz.

## 10. Primer endpoint

Primer endpoint hasta düzeyi üç sınıflı failure-aware balanced accuracy'dir:

`BA_failure-aware = (Recall_normal + Recall_papilledema +
Recall_pseudopapilledema) / 3`.

Her class recall paydası o sınıftaki tüm amaçlanan test hastalarını içerir.
Abstention o hastanın gerçek sınıfı için false negative/hata sayılır.

Primer confusion matrix:

- satırlar: gerçek normal, papilledema, pseudopapilledema;
- sütunlar: tahmin normal, papilledema, pseudopapilledema, abstain;
- boyut: 3×4;
- sıfır hücreler dahil bütün 12 hücre raporlanır.

## 11. Zorunlu sekonder endpointler

Hasta düzeyinde:

- overall ve sınıf-koşullu coverage;
- non-abstained conditional balanced accuracy;
- sınıf başına recall, precision ve F1;
- macro-F1, accuracy ve multiclass MCC;
- sınıf başına one-vs-rest AUROC ve average precision;
- macro AUROC ve macro average precision;
- normal–anormal ayrımı;
- yalnız gerçek hastalık olgularında papilledema–pseudopapilledema ayrımı;
- multiclass NLL ve Brier score;
- top-label ve classwise ECE;
- risk–coverage ve AURC;
- doğru üç-sınıf kararı ile her iki gözde kaynak çalışmada kilitlenmiş
  post-process ROI IoU>=0,50 kriterini en az 4/7 frame'de sağlama koşulunu
  birlikte gerektiren localized diagnostic success. Bu referans-maske bilgisi
  yalnız retrospektif değerlendirmede kullanılır; inference, ROI geçerliliği,
  abstention, checkpoint seçimi veya kalibrasyonu değiştiremez.

Göz düzeyi aynı sınıflandırma çıktıları destekleyici olarak verilir. Frame
düzeyi sınıflandırma gözlemleri bağımsız hasta örnekleri gibi inferans için
kullanılamaz.

Segmentasyon/lokalizasyon sonuçları her tanı sınıfı için ayrı coverage, Dice,
IoU ve failure nedeni dağılımını içerir. Bu analiz upstream'in sınıflar arasında
diferansiyel başarısızlığını görünür kılar; üç-sınıf classifier seçimini test
sonrasında değiştiremez.

## 12. İstatistiksel analiz

- Güven düzeyi %95'tir.
- Resampling birimi hastadır; iki göz ve tüm frameler birlikte korunur.
- Bootstrap sınıfa göre tabakalı 5.000 draw kullanır.
- Primer stratejinin dört modeline ait karşılaştırmalar aynı seed içindeki aynı
  hastalarda paired yapılır; seed'ler havuzlanmaz. Her kontrast için sınıfa göre
  tabakalı 5.000-draw hasta düzeyi BCa bootstrap güven aralığı ve iki yönlü,
  whole-patient sign-flip randomizasyon p-değeri hesaplanır.
- Önceden tanımlı primer aile 6 model çifti × 5 seed = 30 testtir. Holm
  düzeltmesi bu 30 seed-spesifik testin tamamına tek aile olarak uygulanır.
- Diğer çok sayıda endpoint için p-değeri taraması yapılmaz; etki büyüklüğü ve
  güven aralığı önceliklidir.
- Her seed ayrı raporlanır.
- Beş seed aritmetik ortalama ± örneklem SD yalnız tanımlayıcıdır.
- Örtüşen seed test üyelikleri bağımsız tekrarlar sayılmaz ve bağımsız standart
  hata üretmek için birleştirilmez.

Her seed'in testinde yalnız dört papilledema ve dört pseudopapilledema hastası
vardır. Tek hata ilgili sınıf recall'ını 25 yüzde puan değiştirir. Bu kaba çözünürlük,
geniş güven aralıkları ve kalibrasyon belirsizliği açıkça tartışılır.

## 13. Leakage ve fail-closed kontroller

1. Hasta, iki göz ve bütün frameler tek partition'dadır.
2. Classifier train girdileri yalnız hasta-OOF predicted ROI'dir.
3. Validation yalnız checkpoint, calibration ve sekonder uncertainty threshold
   seçimi için kullanılır.
4. Test, hiçbir seçim veya refit kararına girmez.
5. Train-derived normalization/class weights yalnız train'den hesaplanır.
6. ROI reference-standard maskesi classifier inference veya fallback'te
   kullanılmaz.
7. Binary classifier checkpoint/logit/eşik/calibrator import edilmez.
8. Her artifact path, SHA-256 ve boyutla receipt'e bağlanır.
9. Config, protokol, split ve upstream anchor değişirse sonraki aşama durur.
10. Test ROI artifact'ları global sentinel öncesinde okunamaz veya hash'lenemez.

## 14. Validation lock ve test erişim kapısı

Her `4 model × 5 seed × 2 strategy = 40` üç-sınıf classifier için ayrı
validation lock gerekir. Her lock en az şunları bağlar:

- üç-logit classifier checkpoint'i;
- training OOF ve validation ROI import receipt'i;
- monitor durumu ve seçilmiş epoch;
- göz/hasta calibration status ve temperature;
- class order;
- agregasyon ve abstention politikası;
- `test_data_read=false`;
- `test_selection_or_refitting=false`;
- config ve protokol hash'leri.

Global test kapısı şu iki küme tam değilse açılamaz:

- 20/20 development upstream import receipt;
- 40/40 validation lock.

Kapı açıldığında `state/test_access_opened.json` atomik yazılır. Sentinel:

- bütün import receipt ve lock hash'lerini;
- test artifact'larının sentinel öncesi okunmadığını;
- kaynak binary testinin daha önce açıldığını;
- zorunlu exploratory/post-hoc açıklamayı;
- confirmatory claim'in yasak olduğunu

taşır. Sentinel yazıldıktan sonra 20 model×seed deferred-test import receipt'i
oluşturulabilir. Bir receipt, lock veya kaynak artifact sonradan değişirse test
erişimi geçersiz olur.

Zorunlu DAG:

`validate config → verify source anchors/splits → 20 development imports →
40 three-class classifier fits → 40 validation locks → global test sentinel →
20 deferred test imports → test inference/evaluation → immutable reporting`

## 15. CLAIM/TRIPOD+AI uyumlu metadata

`reporting.py` makine-okunur Q1 metadata şemasında şu bölümleri zorunlu kılar:

- çalışma kimliği ve amaç;
- klinik bağlam ve hedef popülasyon;
- veri kaynağı ve sınıf sayıları;
- tanı ve ROI reference standard'ı;
- partitioning ve duplicate audit;
- frozen-upstream kapsamı ve yasak importlar;
- model geliştirme, agregasyon, kalibrasyon ve abstention;
- mimari/başlangıç ağırlığı hash'leri, optimizer, scheduler, batch, epoch,
  augmentation, mixed precision ve determinism ayrıntıları;
- Python/paket, işletim sistemi, CUDA/cuDNN ve GPU ortam snapshot'ı;
- primer/sekonder endpointler;
- istatistik ve belirsizlik;
- önceki test kullanımı;
- external evaluation durumu;
- reproducibility hash'leri;
- sınırlılıklar;
- CLAIM 2024 ve TRIPOD+AI terminolojisi.

“Validation” sözcüğü tuning partition ile karışabildiği için makalede held-out
değerlendirme “internal testing”, bağımsız merkez/zaman kohortu ise “external
testing” olarak adlandırılır.

## 16. Publication output şemaları

En az şu tablolar üretilir:

1. `patient_predictions.csv`
2. `patient_metrics.csv`
3. `patient_confusion_3x4.csv`
4. `classwise_metrics.csv`
5. `calibration_metrics.csv`
6. `risk_coverage.csv`
7. `uncertainty_metrics.csv`
8. `segmentation_by_class.csv`
9. `segmentation_uncertainty.csv`
10. `provenance_artifacts.csv`

Her tablo sabit grain, primary key, sütun tipi, null politikası ve izinli enum
değerine sahiptir. Hasta tahmin tablosu ham ve kalibre üçlü olasılık vektörlerini,
iki gözün evaluability bilgisini, abstention nedenini ve önceki test kullanımının
açıklandığını taşımalıdır. Olasılık vektörleri `1±1e-6` toplamına sahip olmalıdır.

Primer confusion tablosu her model×seed×strategy için 12 hücrenin tamamını
içermelidir. Abstention sütunu çıkarılarak yalnız başarılı vakaların tablosu
primer diye sunulamaz.

Sınıflandırma ve segmentasyon güven aralıkları 5.000 tekrarlı, sınıfa göre
tabakalı bütün-hasta cluster bootstrap ile üretilir. Frame ve gözler bağımsız
örneklem birimi kabul edilmez. BCa tanımlanamadığında percentile fallback,
nedeni ve geçerli tekrar sayısı output satırında görünür olmalıdır.

## 17. Yorum sınırları

Bu çalışma mevcut anatomik segmentasyonu yeniden eğitmeden yeni üç-sınıf
classifierları değerlendiren verimli ve leakage-korumalı bir uzantıdır. Ancak
aynı test hastalarının kaynak ikili çalışmada daha önce görülmüş olması,
tek-merkez veri, küçük papilledema/pseudopapilledema test sayıları, örtüşen seed
holdout'ları ve demografik metadata yokluğu genellenebilirlik iddialarını
sınırlar.

İç çalışmada en iyi görünen model klinik kullanıma hazır kabul edilmez.
Protokolün seçilmiş final sistemi dondurulmalı; yeterli sınıf sayısına sahip,
önceden hiç görülmemiş temporal veya dış merkez kohortta discrimination,
calibration, coverage, failure nedenleri ve klinik yarar birlikte
değerlendirilmelidir.

## 18. Raporlama kaynakları

- Tejani AS ve ark. Checklist for Artificial Intelligence in Medical Imaging
  (CLAIM): 2024 Update. *Radiology: Artificial Intelligence*. 2024;6:e240300.
- Collins GS ve ark. TRIPOD+AI statement: updated guidance for reporting
  clinical prediction models that use regression or machine learning methods.
  *BMJ*. 2024;385:e078378.
- Moons KGM ve ark. PROBAST+AI: updated quality, risk of bias, and
  applicability assessment tool. *BMJ*. 2025;388:e082505.

Bilimsel kural, config veya output şeması değişirse yeni `study_id`, protokol
sürümü ve output namespace gerekir.
