# Strict Predicted-ROI İkili Sınıflandırma Protokolü

Sürüm: 1.2.0
Durum: Kodlama ve ön-kayıt aşaması; klinik eğitim/test henüz çalıştırılmayacak.  
Çalışma tipi: Daha önce incelenmiş iç veri üzerindeki beş tekrarlı hold-out nedeniyle **exploratory/post-hoc internal study**. Doğrulayıcı iddia için protokol dondurulduktan sonra bağımsız dış kohort gerekir.

## 1. Araştırma sorusu ve primer estimand

Primer soru şudur: Dört segmenter kendi model ailesine özgü, ayrı eğitilmiş sıkı predicted-ROI sınıflandırıcısıyla tamamlanmış birer gerçek sistem olduğunda hangisi en yüksek failure-aware tanısal başarıyı sağlar?

Primer karşılaştırma dört farklı segmenterden oluşur:

1. `yolo26`
2. `vit_method2`
3. `emcad`
4. `sam2_unet`

Üç estimand önceden sabittir:

- **E1 — primer `model_specific`:** Her segmenter, kendi aile mimarisine özgü ve yalnız o model×seed için ayrı eğitilen strict-ROI classifier ile eşleştirilir. Aileler sırasıyla YOLO26s, ViT-Method2, PVT-v2-B0 ve SAM2 Hiera-tiny sınıflandırma omurgalarıdır. Bu estimand segmentasyon ile sınıflandırıcı mimarisinin birlikte oluşturduğu tamamlanmış sistem etkisidir; yalnız segmenter etkisi gibi yorumlanmaz.
- **E2 — önceden tanımlı sekonder `standardized_resnet18`:** Her segmenterin ürettiği aynı strict predicted-ROI artifact'ı, model×seed başına ayrı eğitilmiş ortak `torchvision ResNet-18` sınıflandırıcıya verilir. Sınıflandırıcı mimarisini sabit tuttuğu için ROI/segmenter karşılaştırmasının standartlaştırılmış duyarlılık analizidir; primer değildir ve ablation sayılmaz.
- **E3 — doğrudan strateji kontrastı:** Aynı segmenter, seed, test üyeliği ve byte-identical ROI artifact'ında `model_specific − standardized_resnet18` farkıdır. Bu, ROI akışı sabitken sınıflandırıcı stratejisinin ek etkisini ölçer.

Üç estimandın endpoint'i göz düzeyindeki, abstention'ı başarısız karar olarak içeren failure-aware balanced accuracy'dir. Hasta düzeyi sonuç eş-primer değil, her iki strateji için zorunlu sekonder analizdir. “Primer/sekonder” etiketleri iç veri üzerindeki çalışmanın genel exploratory/post-hoc statüsünü değiştirmez.

## 2. Değişmeyen veri ve split akışı

- Görev kontrol `0` ile papilödem+pseudopapilödem `1` ayrımıdır.
- Split birimi hastadır. Aynı hastanın iki gözü ve bütün kesitleri daima aynı partition'dadır.
- Her gözde yedi benzersiz kesit ve her hastada sağ/sol iki göz şarttır.
- Seed'ler tam olarak `17, 42, 2026, 3407, 9103`'tür.
- Önceki çalışmada kullanılan beş hasta-membership CSV'si yeniden örneklenmez. Kaynak dosyalar SHA-256 ile doğrulanır ve yeni, izole `strict_roi_results_4model_v1_2_0/splits` alanına byte-for-byte kopyalanır. Altı-model `strict_roi_results_clean_v1_1_1` koşusundan segmenter checkpoint'i, ROI cache'i veya receipt içe aktarılmaz.
- Train/validation/test kotaları her üç özgün sınıf için sırasıyla `0: 28/10/10`, `1: 13/4/4`, `2: 14/4/4` hastadır. İkili etiket `label_3class > 0` ile elde edilir.
- Manifest ve split hash'lerinden biri değişirse çalışma devam etmez.
- Test partition'ı hiperparametre, eşik, ROI kuralı, kalibrasyon veya checkpoint seçiminde kullanılmaz.

Bu tasarım beş bağımsız kohort anlamına gelmez; seed test kümeleri kısmen aynı hastaları içerebilir. Bu nedenle 5-seed ortalaması ve örneklem standart sapması raporlanır, fakat seed sonuçları bağımsız gözlemlermiş gibi birleştirilerek standart hata hesaplanmaz.

## 3. Segmentasyon eğitimi ve early stopping

Her model/outer-seed çifti için yalnız segmentasyon kaybı kullanılır. Önceki joint sınıflandırma kaybının ağırlığı sıfırdır. Maksimum 60 epoch, patience 10, deterministik seed politikası, AMP, augmentasyon sınırları ve gradient clipping korunur.

Başlangıç ağırlıkları mimarilerin yayımlanmış/eldeki olanaklarına göre önceden kilitlidir: YOLO26 doğrulanmış YOLO26-seg ağırlıklarıyla; EMCAD doğrulanmış PVT-v2-B0 encoder ağırlıklarıyla; SAM2-UNet doğrulanmış Hiera-tiny encoder ağırlıklarıyla başlar. ViT-Method2 segmenter için uyumlu kilitli segmentation checkpoint bulunmadığından random başlar. Segmenter ve classifier başlangıçları ayrı provenance alanlarıyla raporlanır. Eşit olmayan pretraining durumu tamamlanmış sistem kapasitesinin bir parçasıdır; E1 sonucu yalnız mimari veya yalnız segmentasyon etkisi şeklinde yorumlanmaz.

- Kayıp mimariye göre önceden kilitlidir: ViT-Method2, EMCAD ve SAM2-UNet için eşit ağırlıklı BCE + soft Dice; YOLO26 için kendi instance-segmentation başlıklarının doğru supervision almasını sağlayan native YOLO segmentation loss. Bu fark sonuçlarda açıkça raporlanır ve sonradan değiştirilemez.
- Early stopping monitorü: validation göz-düzeyi Dice.
- Eşitlik bozma: validation göz-düzeyi IoU, sonra daha erken epoch.
- En iyi ağırlık geri yüklenir.
- Segmentasyon olasılık eşiği ile component-dominance eşiği outer validation üzerinde ortak grid ile seçilir.
- Grid hedefi, validation göz coverage'ı en az %80 koşuluyla post-process edilmiş göz Dice'ını maksimize etmektir. Burada coverage, en az 4/7 geçerli frame'i olan validation gözlerinin oranıdır. Abstain edilen frame'in segmentasyon tahmini boş kabul edilir; GT ROI'ler non-empty olduğundan Dice=0 alır. Göz Dice'ı yedi frame'in, abstention'ları da içeren aritmetik ortalamasıdır.
- Hiçbir grid adayı %80 coverage sağlamazsa en yüksek coverage, sonra en yüksek post-process göz Dice seçilir ve lock içinde `roi_grid_status=infeasible_fallback` olarak görünür biçimde işaretlenir. Model sessizce elenmez veya test yardımıyla kurtarılmaz.

## 4. Hasta-gruplu cross-fitting

Her iki ROI sınıflandırıcı stratejisinin train girdileri, aynı görüntüyü görmüş bir segmenterden alınamaz. Bu nedenle outer training hastaları sınıfa göre tabakalı beş inner fold'a ayrılır.

Her model/outer-seed için:

1. Bir full outer segmenter yalnız outer train üzerinde eğitilir; outer validation ile early stopping yapılır.
2. Outer train içinde beş inner segmenter kurulur. Her biri dört fold ile, full outer fit'te seçilmiş epoch sayısı kadar eğitilir.
3. Her inner segmenter yalnız eğitiminde görmediği beşinci fold için ROI tahmini üretir.
4. Beş held-out tahmin birleştirilerek her iki classifier stratejisi için tamamen out-of-fold predicted-ROI seti oluşturulur. Bir model×seed×frame için iki strateji aynı ROI tensorunu, hard maskeyi, geçerlilik bayrağını ve abstention nedenini kullanır; her stratejiye ayrı ROI üretimi yapılmaz.
5. Full outer segmenter outer validation ROI'lerini üretir.
6. Test ROI'leri bu aşamada üretilmez.

Hesap yükü ablations hariç önceden sabittir:

- 4 model × 5 seed = 20 full outer segmenter fit,
- 4 model × 5 seed × 5 inner fold = 100 inner OOF segmenter fit,
- toplam 120 segmenter fit,
- E1 `model_specific`: 4 model × 5 seed = 20 classifier fit,
- E2 `standardized_resnet18`: 4 model × 5 seed = 20 classifier fit,
- toplam 40 core classifier fit ve ablation hariç `120 + 40 = 160` learned fit.

Inner held-out fold'un GT maskesi veya sınıf etiketi, o fold'u üreten segmenterin eğitimi ya da epoch seçimi için kullanılmaz.

## 5. Tek predicted-ROI oluşturma algoritması

Algoritma bütün modeller için aynıdır ve classifier'dan önce uygulanır:

1. Full-resolution segmentasyon olasılık haritası elde edilir.
2. Yalnız validation'da seçilmiş eşikle hard maske oluşturulur.
3. 8-bağlı bileşenler çıkarılır. En fazla 64 piksellik iç boşluklar doldurulur; daha büyük boşluklar korunur. Primer analizde morphology closing yoktur (`iterations=0`).
4. Alan alt/üst limitleri yalnız outer-training GT maskelerinden hesaplanır:
   - `A_min = 0.5 × Q01(train GT ROI alan oranı)`
   - `A_max = 1.5 × Q99(train GT ROI alan oranı)`
5. `A_min` altındaki spek/noise parçaları silinir. Hiç aday kalmazsa frame `EMPTY_OR_TINY` olarak abstain olur.
6. Herhangi bir bileşen `A_max` değerini geçerse veya görüntü kenarına temas ederse frame, küçük makul bir parça da bulunsa, `OVERSEGMENTED_OR_EDGE` olarak abstain olur. Büyük FP sessizce atılarak daha küçük parça seçilmez.
7. Tek anatomik aday varsa seçilir.
8. Birden fazla anatomik aday varsa adaylar component içindeki ortalama segmentasyon olasılığıyla puanlanır. En iyi/ikinci skor oranı validation'da kilitlenen dominance eşiğini aşmıyorsa frame `AMBIGUOUS_MULTI` olarak abstain olur. Disconnected adaylar birleştirilmez.
9. Seçilen hard maskenin dışındaki bütün pikseller sabit normalizasyon ortalamasıyla değiştirilir; normalize edildikten sonra dış alan tam sıfırdır.
10. Yalnız seçilen bileşenin sıkı bounding box'ı alınır. Aspect ratio korunarak 224×224 neutral letterbox yapılır. Kutu içinde bile olsa maskenin dışındaki özgün piksel classifier'a verilmez.

GT ROI; classifier girdisini üretmek, başarısız predicted ROI'yi düzeltmek, testte bileşen seçmek veya fallback yapmak için hiçbir zaman kullanılamaz. Full-image/global-feature fallback yoktur.

Bir unit/integration testi, ROI dışındaki bütün özgün pikseller rastgele değiştirildiğinde sabit maske altında her iki classifier stratejisinin logit farkının `1e-6` değerini geçmediğini doğrular. Bu test iki deployable kola ROI dışı bilgi sızmadığının yazılım kanıtıdır.

## 6. ROI sınıflandırıcı eğitimi

- E1 `model_specific` omurgaları config'te model bazında kilitlidir: YOLO26 için YOLO26s; ViT-Method2 için kendi ViT'i; EMCAD için PVT-v2-B0; SAM2-UNet için Hiera-tiny. Her birine yeni iki-logit binary head takılır ve her model×seed için ayrı optimize edilir.
- E1 başlangıçları ayrı ve hash'li olarak tanımlıdır: YOLO26, PVT-v2-B0 ve Hiera-tiny mevcut doğrulanmış aile checkpoint'leriyle; ViT-Method2 uyumlu kilitli checkpoint bulunmadığı için random başlar. Random başlangıç sessiz pretrained fallback ile değiştirilemez.
- E1 trainability politikası aile bazında değiştirilemez biçimde şöyledir:
  - YOLO26: backbone, neck ve yeni binary head trainable'dır. Native detection/segmentation head yalnız neck tensor contract'ını üretmek için forward'a katılır, `requires_grad=false` ve eval modundadır; native segmentation loss ve legacy joint classifier kapalıdır.
  - ViT-Method2: patch embedding, bütün transformer encoder, cross-attention ve yeni binary head trainable'dır; segmentation decoder ve legacy joint classifier kapalıdır.
  - EMCAD: PVT-v2-B0 omurgasının tamamı ve yeni binary head trainable'dır; ImageNet head, EMCAD segmentation decoder ve legacy joint classifier kapalıdır.
  - SAM2-UNet: özgün Hiera-tiny trunk'ın patch/position katmanları ile bütün native attention blokları donuktur. Önce özgün trunk parametreleri dondurulur, sonra her blok trainable `Adapter.prompt_learn` ile sarılır; yalnız bu adapter parametreleri ve yeni binary head optimize edilir. Donuk bloklardan adapter'lara gradient akışı korunur, trunk `no_grad` içine alınmaz. SAM2-UNet segmentation decoder ve legacy joint classifier sınıflandırma graph'ında yoktur.
- ROI üreten fitted segmenter, iki classifier stratejisi eğitilirken daima donuktur. Preflight parameter audit'i trainable/frozen/disabled isimlerini ve gerçek `requires_grad`/optimizer parametre kümelerini config'teki bu politika ile bire bir karşılaştırır.
- E2 `standardized_resnet18`, ImageNet-1K V1 ağırlıklarıyla başlayan ortak ResNet-18'dir; mimari ortak olsa da her segmenter×seed için ağırlıkları ayrı eğitilir.
- YOLO, PVT, Hiera ve ResNet kaynakları/path/SHA-256 ile; random başlangıçlar açık durumla kilitlidir. Çalışma sırasında ağdan otomatik indirme yoktur.
- Her iki strateji yalnız strict masked/cropped ROI tensorunu alır. Hard-mask support, dolayısıyla ROI'nin örtük biçim/alan sinyali görüntüde gözlenebilir; bu nedenle girdi “saf appearance-only” diye adlandırılmaz. Primer kola ayrıca sayısal alan, konum, şekil veya başka explicit geometry vector verilmez.
- Aynı model×seed'de iki stratejinin ROI tensoru ve valid/abstain durumu byte-identical'dır; checkpoint, optimizer state, early stopping, kalibrasyon, eşik ve RNG akışları bağımsızdır. RNG strategy offset'leri E1 için `0`, E2 için `500009`'dur.
- `roi_valid`, segmenterin frame düzeyindeki anatomik/kalite kararının değişmez kaydıdır; gözün classifier eğitimine uygunluğunu temsil etmek için yeniden yazılamaz. Outer-train OOF indeksinden ayrı bir `training_eye_eligibility.csv` oluşturulur. Göz kimliği `patient_id + case_id + side` olup tam yedi benzersiz frame doğrulanır. Yalnız en az 4/7 geçerli predicted ROI'si bulunan gözlerin geçerli frameleri, ayrı `training_optimization_index.csv` üzerinden classifier optimizasyonuna girer. 1–3 geçerli ROI'si bulunan gözlerin gerçek cache yolları korunur fakat bu frameler optimizasyon indeksine alınmaz; 0 geçerli ROI de aynı şekilde coverage failure'dır.
- Ham OOF indeksi, eligibility ledger'ı ve optimizasyon indeksi ayrı SHA-256 alanlarıyla checkpoint, iki strategy training result'u, ortak training manifesti ve receipt'e bağlanır. E1 ile E2 aynı optimizasyon/ledger hash'ini taşımadıkça aşama tamamlanamaz. Resume yalnız aynı code, config, split, ham OOF, eligibility ve stable optimization sıra hash'leriyle mümkündür.
- Optimizasyon AdamW ve cosine-annealing scheduler kullanır. Kayıp hasta-sınıf dengeli olacak ve her gözün bütün geçerli framelerinin toplam ağırlığı eşit olacak biçimde ağırlıklandırılır; daha fazla geçerli ROI üreten göz classifier eğitimini daha fazla domine edemez. Geçerli bir frame için ham ağırlık `1 / (2 × sınıftaki evaluable training göz sayısı × o gözdeki geçerli frame sayısı)` olup minibatch-independent biçimde bütün train setinde ortalama ağırlık 1 olacak şekilde normalize edilir. Sıfır geçerli frame'i olan göz classifier kaybına giremez ve training coverage failure olarak raporlanır.
- Train girdisi inner OOF predicted ROI, validation girdisi full outer segmenterin predicted ROI'sidir. Geçersiz frameler hiçbir stratejide sınıflandırılmaz; göz skoru yalnız geçerli framelerden oluşur.
- Her strategy×model×seed checkpoint'i validation göz AUROC ile, maksimum 60 epoch ve patience 10 kullanılarak bağımsız seçilir. Eşitlik bozucu validation göz NLL'dir.
- Validation'da iki sınıftan evaluable göz yoksa AUROC tanımsızdır; tanımlıysa negatif NLL monitorüne geçilir ve strategy'ye özgü `monitor_status=negative_nll_fallback` kaydedilir.
- Hiç evaluable validation gözü yoksa ilgili strategy fit'i `non_evaluable` olarak durdurulur. Rastgele/default checkpoint, threshold veya sıcaklık uydurulmaz; o strategy'nin test klinik kararları abstain olarak kilitlenir. Bir strategy'nin başarısızlığı diğer strategy'ye fallback yetkisi vermez.
- Validation'da yalnız tek sınıf evaluable olduğu için karar eşiği seçilemiyorsa threshold `unavailable` olur ve testte threshold-temelli kararlar abstain edilir. Test AUROC'u kullanılarak eşik oluşturulmaz.

## 7. Göz ve hasta düzeyi karar

Geçerli frame sayısı `K ≥ 4` ise:

`p_eye = (1/K) × Σ p_frame`

`K < 4` ise göz `ABSTAIN_INSUFFICIENT_FRAMES` olur. E1 ve E2 core analizlerinde maksimum, median veya confidence-weighted ortalama kullanılmaz.

Her iki göz evaluable ise:

`p_patient = (p_right + p_left) / 2`

Tek göz bile abstain ise hasta her iki core stratejide abstain olur. Tek-göz veya öteki classifier strategy'ye fallback yoktur. Göz ve hasta sınıflandırma eşikleri her strategy için validation'da ayrı kilitlenir; testte değiştirilmez.

## 8. Kalibrasyon ve kilit

Ham olasılıklar daima raporlanır. Temperature scaling yalnız outer validation'da ve classifier-strategy/model/seed/raporlama seviyesi için ayrı fit edilir; testte yeniden fit edilmez. Validation koşulları kalibrasyon fit'ini tanımsız yapıyorsa sıcaklık 1 gibi görünürde masum bir default kullanılmaz, `calibration_status=unavailable` kaydedilir. İki validation sınıfı bulunduğu halde yalnız sayısal temperature optimizasyonu başarısızsa klinik karar eşiği ham validation olasılığında seçilip açıkça flag'lenir; kalibre sonuç üretilmez. Validation'da iki sınıf yoksa ham olasılıktan da eşik seçilmez.

Her model/seed validation lock'u en az şunları hash'leriyle içerir:

- full ve beş inner segmenter checkpoint'i,
- E1 ve E2 classifier checkpoint'leri, strategy adları ve ayrı pretrained/random-weight provenance'ları,
- hasta split'i ve OOF fold membership'i,
- segmentasyon eşiği,
- train-only `A_min/A_max`, dominance eşiği ve ROI-grid durumu,
- her strategy için göz ve hasta classifier karar eşikleri,
- her strategy×level için temperature değerleri veya açık `unavailable` durumu,
- iki strategy'nin selected epoch'ları, early-stopping monitorleri ve zorunlu `monitor_status` değerleri,
- her strategy için validation coverage ve failure dağılımı,
- kod, config, protokol ve manifest hash'leri.

Global kapının birimi `4 model × 5 seed = 20` composite model×seed lock'tur. Her composite lock **iki** bağımsız classifier strategy kaydı ve her strategy için göz+hasta olmak üzere iki kalibrasyon/eşik kaydı içermek zorundadır. Böylece core'da 40 strategy lock ve 80 level-specific temperature/threshold lock alanı vardır. Bir strategy eksik veya `unavailable` ise bu durum açıkça kilitlenir; mevcut kol öteki kolun yerine geçmez. Yirmi composite lock'un tamamı geçerli olmadan **hiçbir model veya strategy için test inference açılamaz**.

Her lock doğrulanırken ilgili model preflight'ı ile `train-segmenters → build-rois → train-classifiers → lock` receipt zincirinin tamamı ve bu receipt'lerdeki artifact hash'leri yeniden doğrulanır. İlk test erişiminde 20 unit için bu tam zincirlerin birleşik hash'lerini içeren global ve değiştirilemez bir `test_access_opened.json` yazılır. Core test planı 20 E1 + 20 E2 = 40 system evaluation'dır. Sonradan kod, config, protokol, split, checkpoint, ara receipt veya lock değişirse test devam etmez.

## 9. Test metrikleri

### Segmentasyon ve lokalizasyon

- Frame, göz ve hasta düzeyinde Dice ve IoU,
- pixel sensitivity, specificity ve precision,
- HD95, average symmetric surface distance ve 2-piksel toleranslı surface Dice,
- mutlak/göreli alan hatası ve centroid uzaklığı,
- frame ve göz coverage,
- `EMPTY/TINY`, `OVERSEGMENTED`, `EDGE`, `AMBIGUOUS_MULTI`, `<4/7` failure sayıları,
- sınıf-bazlı coverage.

### Sınıflandırma

E1 ve E2 için ayrı ayrı, zorunlu `classifier_strategy` sütunuyla göz ve hasta düzeyinde:

- AUROC ve average precision,
- sensitivity, specificity, PPV, NPV,
- accuracy, balanced accuracy, F1 ve MCC,
- 2×3 failure-aware confusion matrix: gerçek kontrol/hastalık × tahmin kontrol/hastalık/abstain,
- yalnız non-abstained örnekler için sekonder 2×2 confusion matrix,
- coverage, selective risk, AURC ve risk-coverage eğrisi. Confidence, kilitli karar eşiğine mutlak olasılık uzaklığıdır; eşitlikler stable sıralanır. Gate-abstention'lar en düşük confidence'a eklenir ve hata sayılır. AURC bütün ulaşılabilir coverage adımlarındaki kümülatif riskin ortalamasıdır,
- abstention'ı doğru kabul etmeyen failure-aware accuracy/sensitivity/specificity,
- doğru sınıflandırma + en az 4/7 frame'de GT ROI IoU≥0.5 olarak tanımlanan retrospective localized diagnostic success,
- %0,01–%0,99 arasında 0,01 adımlı decision-curve ve standardized net benefit. Pozitif eylem “papilödem/pseudopapilödem için uzman değerlendirme yolunu tetikleme”dir; abstention model-tetikli müdahale üretmez ve DCA paydası abstention'lar dahil tam hedef kohorttur.

E3 için aynı segmenter×seed×patient kimliklerinde E1−E2 paired farkı, ortak complete intended cohort üzerinde hesaplanır. Localized-success hesabında GT yalnız bütün tahminler dondurulduktan sonra retrospective doğrulama etiketi olarak kullanılır; deployment kararına girmez.

### Kalibrasyon

- Brier score ve negative log-likelihood,
- equal-width ve equal-mass 10-bin ECE. Equal-width aralıklar soldan kapalı/sağdan açıktır; `p=1` son bine girer. Equal-mass için olasılıklar stable sıralanıp en fazla 10 parçaya `array_split` edilir; eşit olasılıklar deterministik hasta sırasına göre farklı binlere bölünebilir. Her bin evaluable örnek oranıyla ağırlıklandırılır,
- calibration intercept ve slope,
- 5.000 patient-cluster draw ile percentile %95 bantlı reliability diagram ve tahmin yoğunluğu,
- ham ve temperature-scaled sonuçların ikisi, her strategy×model×seed×level için ayrı. Olasılık-logit dönüşümünde `epsilon=1e-7`, temperature optimizasyonunda kilitli sınırlar `[0.05, 20]` kullanılır.

Validation/test örneklemi küçük olduğundan calibration slope/intercept ve ECE belirsizliği açıkça vurgulanır; Hosmer–Lemeshow testi primer kalibrasyon kanıtı olarak kullanılmaz.

## 10. Belirsizlik ve model karşılaştırmaları

- %95 güven aralıkları, hastayı cluster olarak yeniden örnekleyen ve iki göz/yedi frame yapısını koruyan 5.000 bootstrap draw ile hesaplanır.
- BCa hesaplanabiliyorsa BCa, değilse percentile interval ve fallback nedeni raporlanır.
- Her seed ayrı verilir; ayrıca beş seed için aritmetik ortalama ± örneklem SD verilir.
- Model farkları aynı seed içindeki aynı hastalar üzerinde paired patient-cluster bootstrap ile effect size ve CI olarak raporlanır; E1, E2 ve E3 için eşleştirme aynı hasta kimliklerini ve complete intended cohort'u korur.
- AUROC için seed-içi DeLong yalnız iki modelde de evaluable ortak altkümede tanımlayıcı sensitivity analizidir. Gözlerin hasta içinde bağımsız olmaması nedeniyle klasik DeLong p-değeri primer kanıt sayılmaz. Eşleşmiş, iki modelde de non-abstained threshold kararlar için McNemar yine sekonderdir; farklı abstention kümelerindeki kapsam sınırlaması açıkça verilir.
- E1 primer karşılaştırma ailesi: `model_specific` tamamlanmış sistemler arasında 6 sırasız çift/seed × 5 seed = **30** göz-düzeyi failure-aware balanced accuracy hipotezi. Etki, config model sırasındaki ilk sistem eksi ikinci sistemdir; CI paired patient-cluster BCa (tanımsızsa percentile), test whole-patient paired label-swap randomization +1'dir; 30 p-değeri tek Holm ailesidir.
- E2 önceden tanımlı standartlaştırılmış karşılaştırma ailesi: ortak ResNet-18 altında 6 segmenter çifti/seed × 5 seed = **30** aynı endpoint hipotezi, E1'den ayrı Holm ailesidir. Bu aile primer diye yeniden etiketlenemez.
- E3 doğrudan within-segmenter strateji ailesi: 4 segmenter/seed × 5 seed = **20** `model_specific − standardized_resnet18` hipotezi. Aynı strict-ROI artifact'ları üzerindeki paired patient-cluster CI ve whole-patient strategy-label-swap testi kullanılır; 20 p-değeri üçüncü, ayrı Holm ailesidir.
- Diğer sınıflandırma endpoint'lerinde her strategy için her metric×level ayrı 30'luk family; kalibrasyon loss/metric'lerinde yine strategy×metric×level başına ayrı 30'luk family tanımlanır. Ablation'larda her test×level, primary `model_specific` strateji içinde her segmenterde bütün ön-tanımlı arm×5 seed karşılaştırmalarından oluşur. Tanımsız p-değeri planlanan family büyüklüğünü küçültmez.
- Yalnız p-değerine göre model seçilmez; estimand, effect size, CI, coverage/abstention ve failure nedenleri birlikte yorumlanır.
- Test hold-out'ları seed'ler arasında örtüştüğünden bütün satırlar bağımsızmış gibi tek havuz test yapılmaz.

## 11. Önceden tanımlı ablation ve kontroller

Ablation'lar 20 E1 primer test ve değiştirilemez **primary-only summary** tamamlandıktan sonra, ayrı checkpoint/validation lock/test makbuzlarıyla çalıştırılır. E2 ortak ResNet-18 bir ablation değil, önceden tanımlı core sekonder comparator'dır. Bütün input/gate/negative-control ablation'ları, teknik olarak tanımlı olduğu her segmenter için E1 `model_specific` classifier ailesiyle yapılır. `lock-ablations` toplam 320 validation lock'unu test görüntüsü açmadan üretir ve tek bir hash'li global lock-set makbuzuyla dondurur. `evaluate-ablations`, bu 320 kilidin ve primer summary makbuzunun değişmediğini doğrulayan ayrı bir test-access sentinel'i yazmadan hiçbir ablation test sonucu açamaz. Ablation sonuçları primer summary'yi değiştirmez ve ayrı, değiştirilemez bir ablation summary üretir:

320 kilidin önceden sabit dökümü şöyledir: whole-image, GT-ROI oracle ve diğer yedi learned arm dahil `9 arm × 4 model-specific classifier ailesi × 5 seed = 180` classifier fit/lock. Whole-image ve GT-oracle girdileri segmenterden bağımsız görünse de E1 classifier mimarileri farklı olduğundan artık tek ortak classifier olarak paylaşılamaz. Yeniden fit gerektirmeyen dört aggregation ve üç minimum-frame varyantı için `(4+3) × 4 × 5 = 140` analitik lock vardır. Böylece ablation fazı **180 classifier fit + 140 no-refit analiz = 320 validation lock ve 320 test evaluation** içerir. Core ile birlikte learned-fit hesabı `120 segmenter + 40 core classifier + 180 ablation classifier = 340`'tır.

1. Whole-image baseline: shortcut kapasitesi; primer yöntem değildir.
2. GT-ROI oracle: kusursuz lokalizasyon üst sınırı; deployable değildir.
3. Unmasked predicted bounding box: bbox içi context etkisi.
4. Yalnız largest component, kalite kapıları yok: rejection kurallarının katkısı.
5. Geometry-only: alan/şekil bilgisinin katkısı.
6. Mask-only: binary maskenin kendi başına sınıflandırma gücü.
7. Background-only negative control: ROI dışında shortcut sinyali.
8. Boyut-eşlenmiş random ROI negative control.
9. Frame aggregation: mean primer; median, maximum ve mask-confidence-weighted mean sekonder.
10. Minimum geçerli frame: 4/7 primer; 1/7 ve 7/7 sensitivity analysis.
11. ROI appearance + geometry: geometrinin incremental katkısı.

Her öğrenilen ablation için train/validation/test ayrımı aynıdır. GT-oracle dışında hiçbir ablation test GT ROI ile inference yapamaz. Ablation sonucu primer model/threshold seçimini geriye dönük değiştiremez.

## 12. CLI aşamaları ve güvenlik kapıları

`preflight`, klinik optimizasyon başlamadan şu kapıları doğrular: CUDA/GPU ve AMP desteği; Python/PyTorch/torchvision/CUDA sürümleri; disk alanı; dört segmenter, dört model-specific classifier initialization'ı ve ortak ResNet-18 ağırlıklarının provenance/hash'leri; her model ve iki strategy için sentetik forward/backward ve çıktı şekli; segmentation-only kayıpta classifier gradient'i bulunmaması; model-specific trainable/frozen/disabled component audit'i; predicted-mask post-process edge-case testleri; iki strategy için ROI dışı piksel invariance testi; byte-identical shared-ROI contract'ı; 3/7 gözün optimizasyondan dışlanıp 4/7 ve 7/7 gözlerin alınmasını ve canonical `roi_valid` değerlerinin değişmemesini doğrulayan sentetik eligibility testi; hasta split/manifest hiyerarşisi ve hash'leri.

Seed 17 dahil hiçbir klinik seed başlatılmadan önce her modelin `verification/{model}.json` artifact'ında iki bağımsız audit round'u `passed` olmalıdır: (1) dual-classifier contract + backward audit; (2) sentetik strict-ROI end-to-end + outside-ROI invariance + 3/7–4/7 training-eligibility audit'i. Her preflight receipt bu artifact'ın hash'ini taşır. Dört modelin preflight receipt'lerinin tamamı doğrulanmadan `train-segmenters` açılamaz. Başarısız veya eksik bir audit klinik fit'i engeller; sentetik test klinik veri inference'ı değildir.

Klinik seed'i segmentasyondan validation lock'a kadar çalıştıran güvenli giriş noktası seed-parametreli launcher'dır. Her çağrı tam olarak bir kilitli seed'i alır, yeni namespace dışından artifact içe aktarmaz ve `evaluate` çağırmadan durur:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_strict_roi_clean_seed.ps1 -Seed 17
powershell -ExecutionPolicy Bypass -File scripts/run_strict_roi_clean_seed.ps1 -Seed 42
```

Seçilen seed'in yeni namespace'inde kısmi artifact zaten varsa launcher örtük olarak devam etmez. Aynı temiz koşunun kesintiden sonra sürdürülmesi açık `-ResumeIncompleteSeed` anahtarı gerektirir; buna rağmen her segmenter checkpoint'i model/outer-seed/inner-fold/fit-index/epoch/code/config kimliğiyle, her classifier checkpoint'i model/seed/strategy/OOF/optimization/eligibility/validation/code/config kimliğiyle yeniden doğrulanır. Bozuk mevcut receipt tamamlanmamış sayılıp üzerine yazılmaz. Başlangıçta fingerprintlenen kaynakların deterministik ZIP snapshot'ı saklanır ve çalışma düzeyindeki exclusive lock farklı seed launcher'larının aynı anda artifact yazmasını engeller. Bu anahtar eski `strict_roi_results` alanından artifact taşımaya izin vermez.

Alttaki aşama komutları denetim/manuel kurtarma içindir; klinik eğitim aşamalarında `--model all --seed <SEED>` açıkça verilmelidir:

```powershell
python -m predicted_roi_study prepare --config predicted_roi_study/config_strict_roi.json
python -m predicted_roi_study preflight --config predicted_roi_study/config_strict_roi.json
python -m predicted_roi_study train-segmenters --config predicted_roi_study/config_strict_roi.json --model all --seed 17
python -m predicted_roi_study build-rois --config predicted_roi_study/config_strict_roi.json --model all --seed 17
python -m predicted_roi_study train-classifiers --config predicted_roi_study/config_strict_roi.json --model all --seed 17
python -m predicted_roi_study lock --config predicted_roi_study/config_strict_roi.json --model all --seed 17
python -m predicted_roi_study evaluate --config predicted_roi_study/config_strict_roi.json
python -m predicted_roi_study summarize --config predicted_roi_study/config_strict_roi.json
python -m predicted_roi_study lock-ablations --config predicted_roi_study/config_strict_roi.json
python -m predicted_roi_study evaluate-ablations --config predicted_roi_study/config_strict_roi.json
python -m predicted_roi_study audit --config predicted_roi_study/config_strict_roi.json
```

`--model` ve `--seed` yalnız önceden tanımlı değerleri kabul eder. `--dry-run` hiçbir fit veya test erişimi başlatmadan hedefleri ve eksik kapıları gösterir. Her tamamlanan aşama artifact hash'lerini içeren atomik bir receipt üretir. Sonraki aşama önceki receipt ve artifact'leri yeniden doğrular.

Zorunlu DAG:

`prepare → preflight [4/4 model, 2 audit round passed] → train-segmenters → build-rois → train-classifiers [20 E1 + 20 E2] → lock → [20/20 composite lock; 40 strategy ve 80 level kaydı] → evaluate [40 core system] → immutable E1 primary + ayrı E2/E3 summarize → lock-ablations [320/320] → evaluate-ablations [global test gate + ayrı ablation summary]`

Buradaki `summarize`, 20 E1 primer model/seed değerlendirmesini değiştirilemez primary summary'ye; 20 E2 değerlendirmeyi ve 20 E3 paired kontrastı açıkça etiketlenmiş ayrı sekonder çıktılara yazar. Her satırda `classifier_strategy` zorunludur. Gelecekteki makale sentezi bu core özetlerle ayrı ablation summary'yi birleştirir; primer tablo E2 veya ablation sonuçlarına bakılarak yeniden yazılmaz.

Primer E1 dört-sistem karşılaştırmasının etki büyüklüğü, aynı test üyeliğinde
kilitli karar ve abstention çıktılarından hesaplanan failure-aware dengeli
doğruluk farkıdır. E2 aynı hesabı ortak ResNet-18 altında yapar; E3 her segmenterde
E1−E2'yi hesaplar. Güven aralığı bütün hastayı yeniden örnekleyen BCa bootstrap
(tanımsızsa percentile), p-değeri tüm hasta çıktısını birlikte yer değiştiren
eşleştirilmiş randomizasyon testidir. Holm aileleri sırasıyla 30, 30 ve 20
hipotez olarak sabittir; tanımsız testler aileyi küçültmez. DeLong ve McNemar
yalnız duyarlılık analizidir. Brier/NLL farkları ortak score-bearing birimlerde,
segmentasyon farkları ise ham ve post-process maskeler için patient-cluster
eşleştirilmiş olarak ayrıca raporlanır; eksik yüzey uzaklığı paydaları açıkça
yazılır.

Hesaplama raporu her `classifier_strategy` için kayıtlı/trainable/frozen/kapalı
legacy parametreleri, checkpoint boyut ve hash'lerini, outer/OOF/classifier eğitim
sürelerini ve ayrıntılı ortam
provenance'ını içerir. Genel FLOP/MAC profiler sayısı dinamik YOLO yolu,
functional işlemler, CPU ROI post-process'i ve abstention'a bağlı classifier
çalışmasını karşılaştırılabilir biçimde kapsamadığı için sayısal olarak verilmez;
bu durum gerekçesiyle birlikte `not_reported` olarak kaydedilir.

Ana niteliksel segmentasyon galerisi `prepare` aşamasında, hiçbir model çıktısı
oluşmadan önce kilitlenir. Her seed ve özgün üç sınıf için bir test karesi;
yalnız test üyeliği, sınıf etiketi ve kararlı kimliklerden hesaplanan SHA-256
sıralamasıyla seçilir ve seed'ler arasında aynı kare tekrar edilmez. Aynı kilitli
kare dört model için yan yana çizilir; Dice,
ROI geçerliliği, güven skoru ve görsel beğeni seçim girdisi olamaz. Ayrı hata
galerisi sonuç-koşulludur (`empty`, `tiny`, `oversegmentation`,
`ambiguous_multi_component`, `edge_touch`), deterministik seçilir ve yalnız
örnekleyici olduğu; hata prevalansı veya performans tahmini olmadığı her
artifact'ta belirtilir.

Bu dosya, JSON config, yeni çalıştırılabilir Python kodu ve adapter'ların import ettiği bütün eski `binary_study/models` Python/YAML kaynaklarının hash'leri receipt zincirine dahildir. Çalışma başladıktan sonra bilimsel bir değişiklik yeni `study_id`, yeni output dizini ve yeni protokol sürümü gerektirir.
