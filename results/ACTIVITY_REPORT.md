# گزارش جامع فعالیت‌های پروژه PTB-XL

**موضوع:** توسعه Teacher و آماده‌سازی خط لوله Knowledge Distillation برای تشخیص MI/Ischemia  
**بازه گزارش:** از آغاز بررسی مستندات تا پایان آزمایش FastTeacher100Hz  
**وضعیت Test:** تا این نقطه باز نشده و ارزیابی نشده است.

## 1. هدف پروژه

هدف، ساخت یک سامانه پژوهشی تکرارپذیر برای تشخیص دودویی «MI یا ایسکمی» در برابر سایر وضعیت‌ها روی PTB-XL و سپس انتقال دانش Teacher به Student است. اولویت فعلی طبق تصمیم کاربر، رساندن Teacher به Accuracy اعتبارسنجی بیش از 90 درصد پیش از شروع آموزش Student و ارزیابی نهایی Test است.

## 2. قواعد روش‌شناسی که رعایت شد

- تعریف کلاس مثبت: وجود هر کد تشخیصی متعلق به کلاس MI یا کد صریح ایسکمی با الگوی `ISC*`.
- Train: foldهای 1 تا 8؛ Validation: fold 9؛ Test: fold 10.
- جداسازی بیماران بین سه بخش کنترل و تأیید شد.
- انتخاب epoch، threshold، TTA و ensemble فقط با Validation انجام شد.
- Test تا نهایی‌شدن Teacher عمداً دست‌نخورده باقی ماند.
- checkpointهای دوره‌ای برای بازیابی پس از قطع برق یا restart طراحی شدند.
- خروجی‌های پژوهشی در قالب CSV، JSON و Markdown ثبت شدند.

## 3. مستندات و ساختار اولیه

در شروع، فایل‌های `README.txt`، راهنمای آموزش Teacher و طرح Student دو-لید با Multi-Level Knowledge Distillation بررسی شدند. مسیر کلی پروژه شامل آماده‌سازی داده، آموزش Teacher دوازده‌لید، آموزش Student دو‌لید، ارزیابی نهایی و تولید بسته پژوهشی بود.

اسکریپت‌های اصلی ایجاد یا تکمیل‌شده عبارت‌اند از:

| اسکریپت | نقش |
|---|---|
| `01_audit_dataset.py` | ممیزی فایل‌ها، رکوردها و برچسب‌ها |
| `02_create_splits.py` | ساخت splitهای patient-disjoint و تعریف برچسب دودویی |
| `03_train_teacher.py` | آموزش Teacher اصلی 500Hz با checkpoint و EMA |
| `04_train_student.py` | آموزش Student baseline/KD؛ اجرای آن تا تثبیت Teacher متوقف ماند |
| `05_evaluate_and_export.py` | ارزیابی نهایی یک‌باره Test و تولید نمودار/جدول |
| `06_build_research_bundle.py` | تجمیع خروجی‌ها در JSON و Markdown |
| `07_optimize_teacher_validation.py` | بهینه‌سازی TTA و threshold فقط روی Validation |
| `08_finetune_teacher.py` | fine-tune با augmentation ضعیف و checkpoint پنج‌دقیقه‌ای |
| `09_fast_teacher_ensemble.py` | ensemble سریع Teacher اصلی و fine-tuned |
| `10_train_fast_teacher_100hz.py` | Teacher سریع 100Hz با ResNet-SE و metadata |

## 4. خط زمانی فعالیت‌ها و تصمیم‌ها

### 4.1 آموزش Teacher اصلی

Teacher اصلی از بلوک‌های ResNet یک‌بعدی، Squeeze-and-Excitation و Transformer روی سیگنال 500Hz استفاده کرد. آموزش 15 epoch کامل شد.

| معیار | نتیجه منتخب |
|---|---:|
| Accuracy حین آموزش | 86.532% |
| Accuracy پس از تنظیم threshold | 87.357% |
| Sensitivity | 82.353% |
| Specificity | 88.423% |
| F1 | 79.208% |
| AUC | 93.096% |

**تصمیم:** چون AUC مناسب ولی Accuracy کمتر از 90 بود، ابتدا از روش‌های کم‌هزینه مانند threshold tuning و TTA استفاده شد.

### 4.2 بهینه‌سازی Validation با TTA و threshold

چهار وضعیت TTA شامل 0، 1، 2 و 4 عبور و threshold با گام 0.001 بررسی شد.

| پیکربندی منتخب | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| TTA=1، threshold=0.591 | 87.448% | 78.235% | 91.617% | 79.522% | 93.072% |

**نتیجه:** بهبود حدود 0.09 واحد درصد بود؛ بنابراین inference optimization به‌تنهایی کافی نبود.

### 4.3 Fine-tune با augmentation ضعیف

augmentation شدید مدل اصلی می‌توانست ویژگی‌های ظریف ECG را تخریب کند. در نتیجه noise، scale و shift ضعیف‌تر، learning rate برابر `1e-5`، label smoothing برابر 0.02 و EMA برابر 0.99 استفاده شد.

| Epoch | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 87.540% | 77.059% | 92.282% | 79.394% | 93.138% |
| 2 | **87.586%** | 75.441% | **93.081%** | 79.106% | 93.131% |
| 3 | 87.494% | 77.059% | 92.216% | 79.334% | **93.162%** |
| 4 | 87.448% | 76.618% | 92.349% | 79.179% | 93.122% |
| 5 | 87.403% | 77.059% | 92.083% | 79.214% | 93.124% |

**تصمیم:** روند ثابت پنج epoch نشان داد که ادامه همین fine-tune احتمالاً بازده اندکی دارد.

### 4.4 Ensemble سریع

پیش‌بینی Teacher اصلی و بهترین fine-tune با TTAهای مختلف، وزن‌های ترکیب و thresholdهای متعدد ترکیب شدند. در مجموع 389 پیکربندی بررسی شد.

| معیار | نتیجه |
|---|---:|
| Accuracy | **87.769%** |
| Sensitivity | 77.500% |
| Specificity | 92.415% |
| F1 | 79.788% |
| AUC | **93.225%** |
| پیکربندی | Original TTA4 + Fine-tuned TTA4 |
| وزن Fine-tuned | 0.90 |
| Threshold | 0.625 |

این بهترین نتیجه کل پروژه تا زمان تنظیم این گزارش است، ولی هدف 90 درصد حاصل نشد.

### 4.5 Teacher سریع 100Hz

برای آزمایش سریع یک خانواده معماری متفاوت، مدل سبک ResNet-SE روی رکورد رسمی 100Hz ساخته شد. میانگین و بیشینه ویژگی‌ها با سن و جنس ادغام شدند. AMP و batch size 128 استفاده شد و DataLoader چهار worker موازی داشت.

| مشخصه | مقدار |
|---|---|
| Train | 17,418 رکورد؛ 5,592 مثبت |
| Validation | 2,183 رکورد؛ 680 مثبت |
| Epoch | 20 |
| Batch size | 128 |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| بهترین epoch | 11 |

| معیار بهترین epoch | نتیجه |
|---|---:|
| Accuracy | 86.257% |
| Sensitivity | 71.618% |
| Specificity | 92.881% |
| F1 | 76.452% |
| AUC | 92.184% |
| Threshold | 0.605 |

epoch بیستم با Accuracy برابر 84.333 درصد تمام شد. افت epochهای پایانی نشان داد این شاخه از بهترین مدل قبلی عبور نکرد.

## 5. مشکلات عملیاتی و راه‌حل‌ها

| مشکل مشاهده‌شده | علت | اقدام انجام‌شده | نتیجه |
|---|---|---|---|
| قطع برق و نیاز به خاموش‌کردن لپ‌تاپ | محدودیت انرژی | checkpoint و ثبت وضعیت مرحله‌ای | امکان ادامه بدون شروع از صفر |
| GPU ظاهراً بیکار | خواندن WFDB و فاصله بین batchها | بررسی پردازش و `nvidia-smi` | مشخص شد استفاده GPU bursty است |
| GPU واقعاً کم‌استفاده در مدل 100Hz | DataLoader تک‌پردازه | چهار worker، prefetch و persistent workers | پس از warm-up، GPU به 100% رسید و سرعت batch بالا رفت |
| فضای C به 0.12GB رسید | فایل‌های موقت و checkpointهای حجیم | پاک‌سازی سه هدف موقت مشخص | فضای آزاد ابتدا به 5.63GB و سپس حدود 47GB رسید |
| درایو D دیده می‌شد ولی قابل دسترسی نبود | محدودیت محیط اجرای فعلی | عدم انتقال checkpoint و پاک‌سازی امن Temp | ادامه ذخیره روی C |
| failure ذخیره snapshot | وجود state حجیم تکراری | snapshot فشرده‌تر و ذخیره اتمیک | جلوگیری از فایل نیمه‌کاره |
| خطای resume در RNG | tensor وضعیت RNG با map_location روی CUDA قرار گرفت | تبدیل صریح RNG به ByteTensor روی CPU | resume از batch 156 موفق شد |
| اجرای ensemble بدون خروجی لحظه‌ای | inference فاقد progress bar و I/O-bound بود | بررسی CPU/process و انتظار تا پایان | 389 ترکیب کامل و نتیجه ثبت شد |
| loss مدل 100Hz در CSV غیرعادی بزرگ بود | جمع epoch به‌جای میانگین ثبت شد | فرمول گزارش اصلاح و میانگین‌های صحیح مستند شد | معیارهای Accuracy/AUC و خود آموزش تحت تأثیر نبودند |
| checkpoint پیشرفت قدیمی‌تر از history نهایی | progress فقط هر پنج دقیقه نوشته می‌شد | اعتبار نهایی از `history.csv` و `report.json` گرفته شد | 20 epoch کامل تأیید شد |

## 6. مدیریت GPU و کارایی

کارت NVIDIA از طریق PyTorch CUDA شناسایی شد. صفر بودن مصرف در بعضی نمونه‌گیری‌ها به معنی توقف نبود؛ عملیات خواندن و decode فایل‌های WFDB روی CPU و دیسک انجام می‌شود و GPU فقط هنگام forward/backward فعال است. در آزمایش 100Hz، پس از فعال‌سازی چهار worker، نمونه مصرف 100 درصد، وضعیت توان P0 و دمای حدود 65 درجه ثبت شد.

برای جلوگیری از برداشت اشتباه، در گزارش‌های بعدی باید هم‌زمان سه شاخص بررسی شوند: پیشرفت batch در ترمینال، وجود پردازش Python و تغییر CPU/I/O، و استفاده/حافظه GPU.

## 7. وضعیت مدل‌ها در یک نگاه

| مدل/روش | بهترین Accuracy Validation | AUC | وضعیت |
|---|---:|---:|---|
| Teacher اصلی + threshold | 87.357% | حدود 93.1% | تکمیل |
| TTA optimization | 87.448% | 93.143% | تکمیل |
| Weak-augmentation fine-tune | 87.586% | 93.162% | تکمیل |
| Ensemble | **87.769%** | **93.225%** | بهترین فعلی |
| FastTeacher100Hz | 86.257% | 92.184% | تکمیل؛ ضعیف‌تر از ensemble |
| Official XResNet1D101 | 85.158% | 91.165% | تکمیل با early stopping؛ ضعیف‌تر از ensemble |

## 8. فایل‌های خروجی مهم

- Teacher اصلی: `C:\ptbxl\project\checkpoints\teacher\teacher_mi_binary_best.pt`
- Fine-tuned Teacher: `C:\ptbxl\results\teacher_optimization\finetune_weak_aug\teacher_finetuned_best.pt`
- گزارش fine-tune: `C:\ptbxl\results\teacher_optimization\finetune_weak_aug\report.json`
- گزارش ensemble: `C:\ptbxl\results\teacher_optimization\fast_ensemble\report.json`
- Teacher سریع: `C:\ptbxl\results\teacher_fast_100hz\teacher_fast_best.pt`
- گزارش Teacher سریع: `C:\ptbxl\results\teacher_fast_100hz\report.json`
- گزارش XResNet رسمی: `C:\ptbxl\results\teacher_official_xresnet101\report.json`
- گزارش توسعه قبلی: `C:\ptbxl\results\TEACHER_DEVELOPMENT_LOG.md/.json`

## 9. تفسیر علمی نتایج

AUC حدود 93 درصد نشان می‌دهد Teacher اصلی رتبه‌بندی مناسبی دارد، اما هم‌پوشانی توزیع احتمال دو کلاس اجازه نمی‌دهد threshold به‌تنهایی Accuracy را به 90 درصد برساند. fine-tune و ensemble تنها بهبود محدود ایجاد کردند. مدل 100Hz سریع‌تر بود، اما کاهش وضوح زمانی و ظرفیت کمتر، نتیجه را بهتر نکرد. بنابراین افزایش epoch یا دست‌کاری بیشتر threshold روی همان Validation احتمال overfitting انتخابی را بالا می‌برد و از نظر پژوهشی توصیه نمی‌شود.

## 10. برنامه ادامه

1. استفاده از معماری رسمی benchmark مانند `xresnet1d101` یا `inception1d` که در ادبیات PTB-XL AUC حدود 0.925 تا 0.937 گزارش کرده‌اند.
2. تطبیق خروجی مدل با برچسب دودویی فعلی، بدون تغییر split و بدون استفاده از Test.
3. checkpoint پنج‌دقیقه‌ای و early stopping بر اساس Accuracy/AUC Validation.
4. در صورت بهبود، ensemble مدل benchmark با Teacher 500Hz فعلی.
5. قفل‌کردن مدل و threshold، سپس یک ارزیابی نهایی روی fold 10.
6. پس از تثبیت Teacher، ادامه Student دو‌لید و Multi-Level Knowledge Distillation.

## 11. ملاحظات بازتولیدپذیری

- seed اصلی 42 است.
- تعداد نمونه‌ها و ماتریس اغتشاش هر مرحله ثبت شده است.
- تمام انتخاب‌ها بر اساس Validation انجام شده‌اند.
- هیچ نتیجه Test در این گزارش وجود ندارد.
- مقدار loss تاریخی FastTeacher100Hz در فایل خام به‌صورت مجموع ثبت شده؛ برای مقایسه باید بر 137 batch تقسیم شود. کد آن برای اجراهای آینده اصلاح شده است.
- قبل از هر ادعای نهایی پایان‌نامه، فایل JSON/CSV متناظر باید منبع عدد باشد، نه خروجی کوتاه ترمینال.

## 12. جمع‌بندی

پروژه از Teacher اولیه 87.357 درصد به بهترین Validation Accuracy برابر 87.769 درصد رسید. این افزایش با TTA، fine-tune و ensemble حاصل شد، اما هدف 90 درصد هنوز محقق نشده است. آزمایش سریع 100Hz نشان داد کاهش محاسبات لزوماً به دقت بهتر منجر نمی‌شود. ادامه منطقی، استفاده از backbone رسمی و قوی‌تر PTB-XL است؛ Test همچنان مهروموم باقی می‌ماند تا اعتبار روش‌شناسی حفظ شود.

## 13. اجرای Official XResNet1D101

برای آزمودن پیشنهاد backbone رسمی، معماری `xresnet1d101` با خروجی دودویی و همان تعریف برچسب و split قبلی اجرا شد. آموزش برای حداکثر 20 epoch برنامه‌ریزی شده بود و پس از شش epoch بدون بهبود، در پایان epoch 13 به‌طور سالم با early stopping خاتمه یافت.

| مشخصه | مقدار |
|---|---:|
| بهترین epoch | 7 |
| Accuracy | 85.158% |
| Sensitivity | 73.529% |
| Specificity | 90.419% |
| F1 | 75.529% |
| AUC | 91.165% |
| Threshold | 0.494 |
| ماتریس اغتشاش | TP=500، FN=180، TN=1359، FP=144 |

epoch 13 با Accuracy برابر 84.563 درصد و AUC برابر 90.988 درصد پایان یافت. کاهش loss آموزش همراه با عدم بهبود پایدار Validation نشانه اشباع/بیش‌برازش بود. بنابراین checkpoint epoch 7 نگه داشته شد، هدف 90 درصد محقق نشد و ensemble قبلی همچنان بهترین انتخاب است. مجموعه Test در این اجرا نیز دست‌نخورده ماند.

## 14. زیرساخت GitHub و اثبات GPU رانر محلی

مخزن عمومی پروژه فقط شامل کد، مستندات و نتایج سبک است؛ دیتاست، checkpoint، محیط مجازی و فایل‌های رانر از Git مستثنا هستند. رانر self-hosted ویندوز با برچسب `ptbxl-local` ثبت شده و workflowهای آن عمداً فقط با اجرای دستی فعال می‌شوند تا pull request عمومی نتواند روی لپ‌تاپ کد اجرا کند.

فایل `runner_gpu_probe.py` یک بار کاری محدود CUDA اجرا می‌کند، نام GPU و نسخه CUDA/PyTorch، تعداد ضرب ماتریسی، زمان اجرا، checksum و شناسه‌های GitHub را در JSON ثبت می‌کند. workflow با نام `GPU Runner Proof` خلاصه و JSON کامل را در log صفحه Action نشان می‌دهد و نسخه ماشین‌خوان آن در `results/GPU_RUNNER_PROOF.json` بایگانی می‌شود.

اجرای end-to-end شماره `32453602059` روی commit `a403ead` موفق شد: GitHub کد را checkout کرد، رانر `LAPTOP-ptbxl-2` کارت `Quadro P1000` را با PyTorch 2.3.1 و CUDA 12.1 فعال کرد، 10,081 ضرب ماتریس 1024×1024 را در 16.688 ثانیه انجام داد و JSON را اعتبارسنجی کرد. تمام مراحل job در 59 ثانیه سبز شدند. پیوند عمومی: `https://github.com/alimirzae/ptbxl-teacher-kd/actions/runs/32453602059`.

در اجرای نخست، مرحله محاسبات GPU موفق بود اما `upload-artifact@v4` به‌علت مشکل شبکه در انتقال فایل 894 بایتی متوقف ماند و session رانر را busy نگه داشت. job به‌صورت force-cancel خاتمه یافت، یک ثبت runner مستقل ساخته شد و workflow طوری اصلاح شد که JSON را مستقیماً در log/summary منتشر و در مخزن بایگانی کند. ثبت قدیمی حذف شد و تنها رانر سالم آنلاین باقی ماند. این رخداد نشان داد مسیر رفت و اجرای GPU صحیح است و وابستگی artifact در شبکه فعلی قابل اتکا نیست.
