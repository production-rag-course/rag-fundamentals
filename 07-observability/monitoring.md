# 3 Pilar Production Visibility untuk Aplikasi LLM

Catatan belajar dari video "Production RAG with LangChain & Vector Databases" (bagian 4:04:41, Three Pillars of Production Visibility).

## 1. Gambaran besar

Satu request ke LLM menghasilkan tiga jenis catatan sekaligus.

```
query masuk
    |
    v
InstrumentedLLM.invoke()          <- pilar 3: pembungkus
    |
    |-- panggil LLM, ukur waktu
    |
    |-- MetricsCollector.record_request()   <- pilar 2: angka (berapa banyak)
    |
    |-- logger.info() -> JSONFormatter      <- pilar 1: cerita (apa yang terjadi)
    |
    |-- @traceable -> LangSmith             <- trace per request
    |
    v
hasil dikembalikan ke user
```

| Pilar | Class | Pertanyaan yang dijawab |
|---|---|---|
| 1. Structured logging | `JSONFormatter` | Apa yang terjadi di request ini? |
| 2. Metrics | `MetricsCollector` | Berapa banyak, seberapa cepat, seberapa sering gagal? |
| 3. Instrumented LLM | `InstrumentedLLM` | Pembungkus yang memasang pilar 1 dan 2 di setiap panggilan LLM |

Ada dua sistem keluaran yang terpisah dan saling melengkapi:

- JSON log keluar ke terminal (stderr). Di production, log ini dikirim ke Datadog, CloudWatch, atau ELK.
- Trace masuk ke LangSmith lewat decorator `@traceable`.

---

## 2. Pilar 1: Structured logging (`JSONFormatter`)

### 2.1 Kenapa perlu

Log biasa:

```
2026-11-24 10:15:02 INFO LLM request completed latency 450ms input 12 output 80
```

Log JSON:

```json
{"level": "INFO", "message": "LLM request completed", "latency_ms": 450, "input_tokens": 12, "output_tokens": 80}
```

Pada log JSON, tiap informasi punya nama (key). Mesin bisa langsung memfilter, misalnya `latency_ms > 1000`.

### 2.2 Alur satu log di Python

```
logger.info("halo")
      |
      v
Python membuat objek "record"
(pesan, level, nama file, nama fungsi, waktu, dll)
      |
      v
handler menerima record
      |
      v
formatter mengubah record menjadi teks
      |
      v
handler menulis teks ke tujuan (terminal)
```

| Pemain | Tugas | Di kode |
|---|---|---|
| Logger | Pintu masuk. Kamu memanggil `.info()` di sini | `logging.getLogger(...)` |
| Handler | Menentukan tujuan tulis | `logging.StreamHandler()` |
| Formatter | Menentukan bentuk teks | `JSONFormatter` |

### 2.3 Kode

```python
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)
```

Python memanggil method `format` otomatis setiap ada log. Kamu tidak memanggilnya sendiri.

| Key di JSON | Sumber | Contoh |
|---|---|---|
| `timestamp` | Waktu sekarang (UTC, format ISO) | `"2026-11-24T03:15:02.123456+00:00"` |
| `level` | Tingkat log | `"INFO"` |
| `message` | Teks pesan | `"LLM request completed"` |
| `module` | File asal log (tanpa `.py`) | `"monitoring"` |
| `function` | Fungsi asal log | `"invoke"` |

Catatan:

- UTC dipakai supaya server di zona waktu berbeda menulis waktu yang sama.
- `getMessage()` dipakai (bukan `record.msg`) karena method ini juga mengisi placeholder. Contoh: `logger.info("user %s login", "budi")` menjadi `"user budi login"`.

### 2.4 `extra_data`

Sisi pemanggil:

```python
self.logger.info(
    "LLM request completed",
    extra={"extra_data": {"latency_ms": 450, "input_tokens": 12}},
)
```

Aturan Python: semua key di dalam `extra={...}` ditempel sebagai atribut ke `record`. Karena key-nya `extra_data`, sekarang ada `record.extra_data`.

Di formatter:

- `hasattr(record, "extra_data")` bertanya: apakah record punya atribut ini?
- Kalau ya, `update()` menggabungkan isinya ke `log_obj`.
- Kalau tidak, langkah itu dilewati.

```python
# Sebelum update
{
    "timestamp": "...",
    "level": "INFO",
    "message": "LLM request completed",
    "module": "monitoring",
    "function": "invoke",
}

# Sesudah update
{
    "timestamp": "...",
    "level": "INFO",
    "message": "LLM request completed",
    "module": "monitoring",
    "function": "invoke",
    "latency_ms": 450,
    "input_tokens": 12,
}
```

Kenapa data dibungkus dengan nama `extra_data`? Supaya key kamu tidak bentrok dengan atribut bawaan `record`. Python melarang `extra` memakai nama seperti `message` atau `name`.

### 2.5 `setup_logging`

```python
def setup_logging():
    logger = logging.getLogger("langgraph_app")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger
```

| Baris | Artinya |
|---|---|
| `getLogger("langgraph_app")` | Ambil logger bernama `langgraph_app`. Nama bebas. |
| `setLevel(logging.INFO)` | Terima log level INFO ke atas. `DEBUG` diabaikan. |
| `StreamHandler()` | Tujuan tulis: terminal. |
| `handler.setFormatter(JSONFormatter())` | Pasang formatter JSON ke handler. |
| `logger.addHandler(handler)` | Pasang handler ke logger. |

Urutan level dari rendah ke tinggi: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

Rantai akhirnya: `logger -> handler -> JSONFormatter -> terminal`.

### 2.6 Contoh yang bisa dijalankan

Simpan sebagai `coba_log.py`:

```python
import json
import logging
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)


logger = logging.getLogger("coba")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)


def kirim_pertanyaan():
    logger.info(
        "request selesai", extra={"extra_data": {"latency_ms": 450, "user": "budi"}}
    )
    logger.warning("jawaban lambat", extra={"extra_data": {"latency_ms": 3200}})
    logger.info("tanpa data tambahan")
    logger.debug("ini tidak muncul")


kirim_pertanyaan()
```

Output (timestamp berbeda tiap kali):

```
{"timestamp": "2026-11-24T03:20:11.482+00:00", "level": "INFO", "message": "request selesai", "module": "coba_log", "function": "kirim_pertanyaan", "latency_ms": 450, "user": "budi"}
{"timestamp": "2026-11-24T03:20:11.483+00:00", "level": "WARNING", "message": "jawaban lambat", "module": "coba_log", "function": "kirim_pertanyaan", "latency_ms": 3200}
{"timestamp": "2026-11-24T03:20:11.483+00:00", "level": "INFO", "message": "tanpa data tambahan", "module": "coba_log", "function": "kirim_pertanyaan"}
```

Yang perlu diperhatikan:

1. Baris 1 punya `latency_ms` dan `user`, karena dikirim lewat `extra_data`.
2. Baris 2 hanya punya `latency_ms`. Field tambahan bisa berbeda tiap log.
3. Baris 3 tidak punya field tambahan, karena `hasattr` bernilai false.
4. Log `debug` tidak muncul, karena level logger adalah INFO.

### 2.7 Dipakai untuk apa

Di Datadog atau CloudWatch kamu bisa mencari:

- `level = WARNING` untuk semua peringatan.
- `latency_ms > 1000` untuk semua request lambat.
- `user = budi` untuk semua aktivitas satu user.

### 2.8 Latihan

1. Apa yang terjadi kalau `setLevel(logging.INFO)` diganti `logging.DEBUG`?
2. Tambahkan `"token_total": 92` ke salah satu `extra_data`. Apakah muncul di output?

---

## 3. Pilar 2: Metrics (`MetricsCollector`)

### 3.1 Log vs metrics

Tiga request masuk:

| Request | Latency | Error? |
|---|---|---|
| 1 | 100 ms | tidak |
| 2 | 300 ms | tidak |
| 3 | 200 ms | ya |

- Log mencatat tiga kejadian terpisah.
- Metrics merangkum: 3 request, 1 error, rata-rata latency 200 ms.

Log berguna untuk menyelidiki satu kasus. Metrics berguna untuk melihat tren di dashboard dan membuat alarm.

### 3.2 Delapan counter

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
```

| Counter | Menghitung apa |
|---|---|
| `requests_total` | Jumlah request |
| `errors_total` | Jumlah request yang gagal |
| `latency_sum` | Total waktu semua request (ms) |
| `latency_count` | Berapa kali latency dicatat |
| `tokens_input` | Total token masuk |
| `tokens_output` | Total token keluar |
| `cache_hits` | Jumlah jawaban yang diambil dari cache |
| `cache_misses` | Jumlah jawaban yang tidak ada di cache |

### 3.3 `record_request`

Method ini dipanggil sekali per request.

```python
def record_request(
    self, latency_ms, input_tokens, output_tokens, error=False, cache_hit=False
):
    # counter yang selalu naik
    self.metrics["requests_total"] += 1
    self.metrics["latency_sum"] += latency_ms
    self.metrics["latency_count"] += 1
    self.metrics["tokens_input"] += input_tokens
    self.metrics["tokens_output"] += output_tokens

    # counter bersyarat
    if error:
        self.metrics["errors_total"] += 1

    if cache_hit:
        self.metrics["cache_hits"] += 1
    else:
        self.metrics["cache_misses"] += 1
```

`cache_hits` dan `cache_misses` selalu salah satu yang naik. `errors_total` hanya naik saat terjadi error.

### 3.4 Contoh jalan

```python
m = MetricsCollector()
m.record_request(latency_ms=100, input_tokens=10, output_tokens=50)
m.record_request(latency_ms=300, input_tokens=20, output_tokens=80, cache_hit=True)
m.record_request(latency_ms=200, input_tokens=0, output_tokens=0, error=True)
```

| Counter | Awal | Setelah #1 | Setelah #2 | Setelah #3 |
|---|---|---|---|---|
| `requests_total` | 0 | 1 | 2 | 3 |
| `errors_total` | 0 | 0 | 0 | 1 |
| `latency_sum` | 0 | 100 | 400 | 600 |
| `latency_count` | 0 | 1 | 2 | 3 |
| `tokens_input` | 0 | 10 | 30 | 30 |
| `tokens_output` | 0 | 50 | 130 | 130 |
| `cache_hits` | 0 | 0 | 1 | 1 |
| `cache_misses` | 0 | 1 | 1 | 2 |

### 3.5 Kenapa `latency_sum` dan `latency_count` terpisah

Pembicara menyebutnya "you can't average averages".

Rata-rata setelah request #3 adalah `600 / 3 = 200 ms`. Kalau kamu hanya menyimpan rata-rata lalu merata-ratakannya lagi dengan nilai baru, hasilnya salah. Rata-rata 150 yang berasal dari 1 request punya bobot berbeda dengan rata-rata 150 dari 1000 request. Total dan jumlah data tidak punya masalah ini.

Keuntungan tambahan: data dari beberapa server mudah digabung. Jumlahkan semua `latency_sum`, jumlahkan semua `latency_count`, lalu bagi.

### 3.6 `get_summary`

Tiga perhitungan dengan pola sama: bagi, kecuali pembaginya nol.

```python
avg_latency = (
    self.metrics["latency_sum"] / self.metrics["latency_count"]
    if self.metrics["latency_count"] > 0
    else 0
)
```

Dibaca: "kalau `latency_count` lebih dari 0, hitung sum dibagi count. Kalau tidak, hasilnya 0." Tanpa pengecekan ini, `0 / 0` melempar `ZeroDivisionError`.

| Metrik | Rumus | Hasil pada contoh |
|---|---|---|
| `avg_latency` | `latency_sum / latency_count` | 600 / 3 = 200.0 |
| `error_rate` | `errors_total / requests_total` | 1 / 3 = 0.3333 |
| `cache_hit_rate` | `cache_hits / (cache_hits + cache_misses)` | 1 / 3 = 0.3333 |

Format keluaran:

- `f"{error_rate:.2%}"` mengubah `0.3333` menjadi `"33.33%"`.
- `round(avg_latency, 2)` membulatkan ke dua desimal.

### 3.7 Contoh yang bisa dijalankan

Simpan sebagai `coba_metrics.py`:

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def record_request(
        self, latency_ms, input_tokens, output_tokens, error=False, cache_hit=False
    ):
        self.metrics["requests_total"] += 1
        self.metrics["latency_sum"] += latency_ms
        self.metrics["latency_count"] += 1
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens
        if error:
            self.metrics["errors_total"] += 1
        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def get_summary(self):
        m = self.metrics
        avg_latency = (
            m["latency_sum"] / m["latency_count"] if m["latency_count"] > 0 else 0
        )
        error_rate = (
            m["errors_total"] / m["requests_total"] if m["requests_total"] > 0 else 0
        )
        total_cache = m["cache_hits"] + m["cache_misses"]
        cache_hit_rate = m["cache_hits"] / total_cache if total_cache > 0 else 0
        return {
            "total_requests": m["requests_total"],
            "total_errors": m["errors_total"],
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "total_input_tokens": m["tokens_input"],
            "total_output_tokens": m["tokens_output"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
        }


m = MetricsCollector()
print("Sebelum ada request:", m.get_summary())

m.record_request(latency_ms=100, input_tokens=10, output_tokens=50)
m.record_request(latency_ms=300, input_tokens=20, output_tokens=80, cache_hit=True)
m.record_request(latency_ms=200, input_tokens=0, output_tokens=0, error=True)

print("Setelah 3 request:")
for key, value in m.get_summary().items():
    print(f"  {key}: {value}")
```

Output:

```
Sebelum ada request: {'total_requests': 0, 'total_errors': 0, 'error_rate': '0.00%', 'avg_latency_ms': 0, 'total_input_tokens': 0, 'total_output_tokens': 0, 'cache_hit_rate': '0.00%'}
Setelah 3 request:
  total_requests: 3
  total_errors: 1
  error_rate: 33.33%
  avg_latency_ms: 200.0
  total_input_tokens: 30
  total_output_tokens: 130
  cache_hit_rate: 33.33%
```

Baris pertama membuktikan pengecekan `> 0` bekerja.

### 3.8 Batasan

- Data hanya ada di memori. Setelah program restart, semua counter kembali ke 0. Di production dipakai Prometheus.
- Tidak aman untuk banyak thread sekaligus. Dua request bersamaan bisa saling menimpa operasi `+=`.
- Rata-rata dihitung sejak program mulai. Kamu tidak bisa bertanya "rata-rata 5 menit terakhir".

### 3.9 Latihan

1. Tambahkan request keempat: `latency_ms=400`, tanpa error, tanpa cache. Berapa `avg_latency_ms` dan `cache_hit_rate` yang baru?
2. Saat error, kenapa kode asli mengirim `input_tokens=0` dan `output_tokens=0`?

---

## 4. Pilar 3: Instrumented LLM (`InstrumentedLLM`)

### 4.1 Peran

Kalau kamu memanggil logger dan metrics manual di setiap tempat yang memakai LLM, suatu saat ada yang lupa. Solusinya: bungkus LLM dalam satu class.

```
Tanpa pembungkus:
  kode A -> LLM          (lupa catat)
  kode B -> LLM + catat
  kode C -> LLM          (lupa catat)

Dengan pembungkus:
  kode A -> InstrumentedLLM -> LLM + catat
  kode B -> InstrumentedLLM -> LLM + catat
  kode C -> InstrumentedLLM -> LLM + catat
```

### 4.2 `__init__`

```python
class InstrumentedLLM:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.metrics = MetricsCollector()
        self.logger = setup_logging()
```

| Atribut | Isi | Berasal dari |
|---|---|---|
| `self.llm` | Klien LLM asli | `ChatOpenAI` |
| `self.metrics` | 8 counter, semua 0 | Pilar 2 |
| `self.logger` | Logger JSON | Pilar 1 |

### 4.3 `@traceable`

```python
@traceable(name="instrumented_invoke")
def invoke(self, query: str) -> str:
```

Decorator dari LangSmith ini merekam input, output, dan waktu tiap panggilan `invoke()`. Hasilnya tampil di dashboard LangSmith dengan nama `instrumented_invoke`. Tidak perlu kode tambahan.

### 4.4 Jalur sukses

Contoh: `query = "What is Python?"`.

**Langkah 1. Mulai stopwatch.**

```python
start_time = time.time()
```

**Langkah 2. Panggil LLM.**

```python
response = self.llm.invoke(query)
result = response.content
```

**Langkah 3. Estimasi token.**

```python
input_tokens = len(query.split()) * 4 // 3
output_tokens = len(result.split()) * 4 // 3
```

Rumus: jumlah kata dikali 4, dibagi 3 (pembagian bulat). Perkiraan kasar: 1 kata sekitar 1,33 token.

| | Teks | Kata | Hitungan | Token |
|---|---|---|---|---|
| Input | `What is Python?` | 3 | 3 × 4 // 3 | 4 |
| Output | `Python is a programming language used for web, data, and automation.` | 12 | 12 × 4 // 3 | 16 |

**Langkah 4. Catat ke metrics (pilar 2).**

```python
self.metrics.record_request(
    latency_ms=(time.time() - start_time) * 1000,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    error=False,
    cache_hit=False,
)
```

Selisih waktu dalam detik dikali 1000 menjadi milidetik.

**Langkah 5. Tulis log (pilar 1).**

```python
self.logger.info(
    "LLM request completed",
    extra={
        "extra_data": {
            "latency_ms": (time.time() - start_time) * 1000,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    },
)
```

**Langkah 6. Kembalikan jawaban.**

```python
return result
```

Pemanggil hanya menerima teks jawaban. Dia tidak tahu ada pencatatan di belakang layar.

### 4.5 Jalur gagal

Kalau `self.llm.invoke(query)` melempar exception, Python langsung lompat ke blok `except`:

```python
except Exception as e:
    error = True
    self.metrics.record_request(
        latency_ms=(time.time() - start_time) * 1000,
        input_tokens=0,
        output_tokens=0,
        error=True,
        cache_hit=False,
    )
    self.logger.error(
        f"LLM request failed: {e}", extra={"extra_data": {"error": str(e)}}
    )
    raise
```

1. Metrics dicatat dengan `error=True`, jadi `errors_total` naik.
2. Token diisi 0, karena tidak ada jawaban untuk dihitung.
3. Log level `ERROR` ditulis dengan pesan error.
4. `raise` tanpa argumen melempar ulang exception yang sama.

`raise` wajib ada. Pembungkus ini hanya bertugas mencatat dan tidak boleh menelan error. Tanpa `raise`, kode pemanggil mengira request sukses.

### 4.6 Alur lengkap

```
invoke("What is Python?")
    |
    |-- start_time = sekarang
    |
    |-- try:
    |     panggil LLM ----------------+
    |     |                           |
    |   sukses                      gagal
    |     |                           |
    |   hitung token            metrics: error=True, token=0
    |   metrics: error=False    logger.error(...)
    |   logger.info(...)        raise (lempar ulang)
    |   return jawaban
    |
    v
(@traceable merekam semuanya ke LangSmith)
```

### 4.7 Contoh yang bisa dijalankan tanpa API key

LLM diganti dengan LLM palsu. Simpan sebagai `coba_instrumented.py` di folder yang sama dengan file monitoring. Kalau nama file monitoring bukan `monitoring.py`, ubah baris import.

```python
import time
from monitoring import setup_logging, MetricsCollector


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def invoke(self, query):
        time.sleep(0.2)
        if "error" in query:
            raise RuntimeError("API timeout")
        return FakeResponse(
            "Python is a programming language used for web, data, and automation."
        )


class WrappedLLM:
    def __init__(self):
        self.llm = FakeLLM()
        self.metrics = MetricsCollector()
        self.logger = setup_logging()

    def invoke(self, query):
        start_time = time.time()
        try:
            result = self.llm.invoke(query).content
            input_tokens = len(query.split()) * 4 // 3
            output_tokens = len(result.split()) * 4 // 3
            latency = (time.time() - start_time) * 1000

            self.metrics.record_request(latency, input_tokens, output_tokens)
            self.logger.info(
                "LLM request completed",
                extra={
                    "extra_data": {
                        "latency_ms": round(latency, 1),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                },
            )
            return result
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.metrics.record_request(latency, 0, 0, error=True)
            self.logger.error(
                f"LLM request failed: {e}",
                extra={"extra_data": {"error": str(e)}},
            )
            raise


llm = WrappedLLM()

for q in ["What is Python?", "Explain machine learning.", "trigger error"]:
    try:
        llm.invoke(q)
    except RuntimeError:
        pass

print("\nRingkasan metrics:")
for key, value in llm.metrics.get_summary().items():
    print(f"  {key}: {value}")
```

Perbedaan dari kode asli: tidak ada `@traceable`, latency dihitung sekali, dan loop menangkap error supaya program tidak berhenti.

Output (timestamp dan latency sedikit berbeda):

```
{"timestamp": "2026-11-24T03:40:01.201+00:00", "level": "INFO", "message": "LLM request completed", "module": "coba_instrumented", "function": "invoke", "latency_ms": 200.4, "input_tokens": 4, "output_tokens": 16}
{"timestamp": "2026-11-24T03:40:01.402+00:00", "level": "INFO", "message": "LLM request completed", "module": "coba_instrumented", "function": "invoke", "latency_ms": 200.3, "input_tokens": 4, "output_tokens": 16}
{"timestamp": "2026-11-24T03:40:01.603+00:00", "level": "ERROR", "message": "LLM request failed: API timeout", "module": "coba_instrumented", "function": "invoke", "error": "API timeout"}

Ringkasan metrics:
  total_requests: 3
  total_errors: 1
  error_rate: 33.33%
  avg_latency_ms: 200.3
  total_input_tokens: 8
  total_output_tokens: 32
  cache_hit_rate: 0.00%
```

Cara membaca:

- Dua baris `INFO` untuk dua request sukses, masing-masing 4 token input dan 16 token output.
- Satu baris `ERROR` untuk `trigger error`. Tidak ada field token, karena jalur gagal tidak menulisnya.
- Total token input 8 (4 + 4 + 0). Total token output 32 (16 + 16 + 0).
- `cache_hit_rate` 0%, karena class ini selalu mengirim `cache_hit=False`.
- Log keluar lewat stderr, jadi urutannya bisa tercampur dengan hasil `print` di terminal tertentu.

### 4.8 Kaitan dengan `demo_monitoring()` di video

Fungsi demo di kode asli melakukan tiga hal:

1. Membuat satu `InstrumentedLLM()`.
2. Memanggil `invoke()` untuk 3 pertanyaan. Tiap panggilan menghasilkan satu log JSON dan satu catatan metrics.
3. Memanggil `llm.metrics.get_summary()` di akhir untuk mencetak ringkasan.

Di video latency terlihat ribuan milidetik karena LLM yang dipakai sungguhan.

### 4.9 Latihan

1. Tambahkan query keempat `"another error"` pada contoh di atas. Berapa `error_rate` dan `avg_latency_ms` setelahnya?
2. Hapus baris `raise` di blok `except`. Apa yang berubah di output? Kenapa itu berbahaya?
3. Buat `WrappedLLM()` dua kali di file yang sama, lalu panggil `invoke` di salah satunya. Berapa baris log yang muncul?

---

## 5. Kelemahan kode di video

Kode ini ditulis untuk demo. Untuk production, perhatikan hal berikut.

| Masalah | Penjelasan | Perbaikan |
|---|---|---|
| Handler ganda | `setup_logging()` dipanggil di `__init__`. Tiap `InstrumentedLLM()` baru menambah handler, sehingga log muncul dobel. | Cek `if not logger.handlers:` sebelum `addHandler`, atau panggil `setup_logging()` sekali di luar class. |
| Token hanya estimasi | Rumus `kata * 4 // 3` hanya perkiraan. | Pakai `response.usage_metadata["input_tokens"]` dan `["output_tokens"]`. |
| Metrics hilang saat restart | Data hanya di memori. | Pakai Prometheus. |
| `cache_hit` selalu `False` | Belum ada cache di class ini. | Sambungkan dengan layer cache. |
| Variabel `error` tidak terpakai | `error = False` di awal `invoke()` tidak dibaca. | Hapus. |
| Latency dihitung dua kali | Satu untuk metrics, satu untuk log, jadi angkanya sedikit berbeda. | Hitung sekali ke variabel, pakai di kedua tempat. |
| Tidak thread-safe | Operasi `+=` bisa saling menimpa saat request bersamaan. | Pakai lock, atau library metrics. |

---

## 6. Ringkasan satu halaman

| | Pilar 1 | Pilar 2 | Pilar 3 |
|---|---|---|---|
| Nama | Structured logging | Metrics | Instrumented LLM |
| Class | `JSONFormatter` | `MetricsCollector` | `InstrumentedLLM` |
| Jawab | Apa yang terjadi? | Berapa banyak? | Siapa yang memasang keduanya? |
| Bentuk data | Satu baris JSON per kejadian | 8 counter di dictionary | Pembungkus di sekitar LLM |
| Tujuan | Datadog, CloudWatch, ELK | Dashboard, alarm | Semua panggilan LLM tercatat otomatis |
| Titik kunci | `extra_data` + `hasattr` + `update` | `latency_sum` dan `latency_count` terpisah | `try/except` + `raise` |