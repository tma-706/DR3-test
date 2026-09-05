# Local user-files-only evaluation

Pipeline này chấm report local bằng đúng 5 metric DR3:

- **IR**: Information Recall, dùng gold_insights_from_source.json.
- **CC**: Citation Coverage, dùng cơ chế extract/match/fallback/tính điểm gốc của DR3,
  nhưng danh sách source bắt buộc chỉ gồm official user files.
- **FA**: Factual Accuracy, dùng claim extraction, source verification, prompt và công
  thức tính điểm gốc; chỉ kiểm tra claim có explicit citation với user file thật.
- **IF**: Instruction Following, dùng checklist.json.
- **DQ**: Depth Quality, dùng query gốc trong query.jsonl.

Pipeline không dùng DR3-Agent, sandbox corpus, useful_search, long_context, web, và không tự sinh gold/checklist.

## Cấu trúc dữ liệu

~~~text
local_eval/
  datasets/
    query.jsonl
    008/
      <official user files>
  ground_truth/
    008/
      checklist.json
      gold_insights_from_source.json
  results/
    008/
      final_report.pdf
~~~

Task ID được normalize về 3 chữ số, nên **--task 8** và **--task 008** là một task.

## Cấu hình

Trong .env:

~~~dotenv
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=qwen/qwen3.7-flash
OPENROUTER_VISION_MODEL=qwen/qwen2.5-vl-32b-instruct
EVAL_TEMPERATURE=0
~~~

OPENROUTER_MODEL chấm 5 metric. OPENROUTER_VISION_MODEL chỉ được gọi khi PDF có ảnh/chart/flowchart lớn và mang nội dung. Logo, header, footer và decoration nhỏ bị loại bằng heuristic local trước khi gọi vision.

Lưu ý: khi chạy evaluation thật, nội dung report và phần user file cần xác minh sẽ được gửi tới model đã cấu hình trên OpenRouter.

## Lệnh chạy

Kiểm tra task mà không gọi API hoặc ghi output:

~~~powershell
uv run python local_eval_runner.py --task 008 --dry-run
~~~

Chạy một task:

~~~powershell
uv run python local_eval_runner.py --task 008
~~~

Khi một task có nhiều report hoặc cần chọn format cụ thể:

~~~powershell
uv run python local_eval_runner.py --task 012 --report local_eval/results/012/final_report.tex
~~~

Chạy nhiều task:

~~~powershell
uv run python local_eval_runner.py --task 008 --task 012 --workers 1
~~~

Chạy mọi task đang có report:

~~~powershell
uv run python local_eval_runner.py --all --workers 1
~~~

Mặc định runner tái dùng preprocess và metric JSON đã hoàn tất. Dùng **--overwrite** để chấm lại:

~~~powershell
uv run python local_eval_runner.py --task 012 --overwrite
~~~

**--workers 1** là mặc định an toàn. Chỉ tăng workers khi quota/rate limit OpenRouter cho phép.

## Preprocess report

- Markdown, text và LaTeX (`.md`, `.markdown`, `.txt`, `.text`, `.tex`) được đọc toàn bộ.
- PDF được trích toàn bộ text theo page/block và thêm page boundary dạng HTML comment.
- Không có giới hạn first-N characters.
- Visual substantive được render riêng và transcription bằng vision; kết quả được chèn đúng vị trí trang.
- Không có visual substantive thì không gọi vision.
- Metadata ghi rõ page nào là candidate, page nào đã dùng vision và model nào được dùng.

## Output

Mỗi report ghi theo format vào `eval_result/<task>/<format>/`, ví dụ
`eval_result/012/pdf/` hoặc `eval_result/012/tex/`:

~~~text
report_for_eval.md
preprocess_metadata.json
eval_information_recall.json
eval_citation_coverage.json
eval_factual_accuracy.json
eval_format_compliance.json
eval_depth_quality.json
scores.json
summary.json
~~~

Mỗi metric có status, score, details và thông tin cache. Quy ước:

- Không có user file nào được cơ chế CC gốc match: CC = 0, status = success.
- Không có explicit cited claim-source pair: FA = 0, status = success.
- Thiếu file/query/gold, lỗi parse hoặc lỗi API/auth: score = null, status = error.
- Zero hợp lệ không bị đổi thành lỗi hoặc bị bỏ khỏi summary.

Runner in bảng IR/CC/FA/IF/DQ sau khi hoàn tất.

## Citation gián tiếp trong report

Lớp local hỗ trợ resolve citation đánh số mà không phụ thuộc tên heading của phần
tài liệu tham khảo:

~~~text
Một claim trong report [1].

[1]: HUD-sec8-FY25.pdf
~~~

Tên file phải khớp chính xác basename của một official user file trong dataset.
Các dạng `[1, 2]`, `[1-3]`, Markdown footnote `[^1]`, danh sách `1. file.pdf`,
và LaTeX `\cite{key}` + `\bibitem{key}` cũng được hỗ trợ. URL được phân loại
riêng; URL kết thúc bằng `.pdf` không tự động được coi là user file. Nếu một entry
có cả basename user file và URL, FA chỉ dùng basename để kiểm tra supplied data.

CC vẫn giữ nguyên điểm và fallback của DR3 gốc. Resolver chỉ bổ sung
`explicitly_cited` và chuyển citation đã resolve sang dạng `[Doc: file]` trong bộ
nhớ trước khi gọi FA; report gốc không bị sửa. File chỉ xuất hiện trong bibliography
nhưng không được cite trong nội dung không đủ để kích hoạt FA.

## Test local

~~~powershell
uv run pytest tests/test_local_eval.py -q
uv run python local_eval_runner.py --task 008 --dry-run
uv run python local_eval_runner.py --all --dry-run
~~~

Prompts/rubrics và source code của evaluator CC/FA gốc được giữ nguyên. Adapter local chỉ
thay scope source thành official user files, nối API client đã cấu hình, bảo vệ zero hợp lệ,
và tiền xử lý report/source PDF khi cần.
