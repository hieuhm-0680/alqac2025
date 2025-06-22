# Competition name: ALQAC2025
This is the overview of the ALQAC2025 competition.

```
As an associated event of KSE 2025, we are happy to announce the 5th Automated Legal Question Answering Competition (ALQAC 2025). ALQAC 2025 includes two tasks: 

Legal Document Retrieval

Legal Question Answering

For the competition, we introduce a new Legal Question Answering dataset – a manually annotated dataset based on well-known statute laws in the Vietnamese language. Through the competition, we aim to develop a research community on legal support systems. While the data is in Vietnamese, we extend a warm invitation to international teams to join us in uncovering the potential of multilingual methods and models.
```

## Task

### Task 1: Legal Document Retrieval
Task 1’s goal is to return the article(s) that are related to a given question. The article(s) are considered “relevant” to a question iff the question can be answered using the article(s).

The training data is in JSON format as follows:

```
[
    {
        "question_id": "DS-101",
        "question_type": "Đúng/Sai",
        "text": "Cơ sở điện ảnh phát hành phim phải chịu trách nhiệm trước pháp luật về nội dung phim phát hành, đúng hay sai?",
        "relevant_articles": [
            {
                "law_id": "05/2022/QH15",
                "article_id": "15"
            }
        ]
    }
]
```

The test data is in JSON format as follows:

```
[
    {
        "question_id": "DS-1",
        "question_type": "Đúng/Sai",
        "text": "Phim đã được Bộ Văn hóa, Thể thao và Du lịch, Ủy ban nhân dân cấp tỉnh cấp giấy phép phân loại phim sẽ có giá trị trên toàn quốc, đúng hay sai?"
    }
]
```

The system should retrieve all the relevant articles. Please see the Submission Details section below for the format of the submissions.

Note that “relevant_articles”  is the list of all relevant articles to the questions.

The evaluation methods are precision, recall, and F2-measure as follows:
$$
\begin{align*}
\text{precision}_i &= \frac{\text{the number of correctly retrieved articles of question}~ i-th}{\text{the number of retrieved articles of question}~ i-th} \\
\text{recall}_i &= \frac{\text{the number of correctly retrieved articles of question}~ i-th}{\text{the number of relevant articles of question}~ i-th} \\
\text{F2}_i &= \frac{5 \cdot \text{precision}_i \cdot \text{recall}_i}{4 \cdot \text{precision}_i + \text{recall}_i} \\
\text{F2} &= \text{average}(\text{F2}_i) 
\end{align*}
$$

In addition to the above evaluation measures, ordinal information retrieval measures such as Mean Average Precision and R-precision can be used for discussing the characteristics of the submission results. The macro-average F2-measure is the principal measure for Task 1.

### Task 2: Legal Question Answering
Given a legal question, the goal is to answer the question. In ALQAC 2024, there are three types of questions:
1. True/False questions (Câu hỏi Đúng/Sai). Here is a training example:

```
[
    {
        "question_id": "DS-101",
        "question_type": "Đúng/Sai",
        "text": "Cơ sở điện ảnh phát hành phim phải chịu trách nhiệm trước pháp luật về nội dung phim phát hành, đúng hay sai?",
        "relevant_articles": [
            {
                "law_id": "05/2022/QH15",
                "article_id": "15"
            }
        ],
	"answer": "Đúng"
    }
]
```
For the True/False questions, the answer must be "Đúng" or "Sai".

2. Multiple-choice questions (Câu hỏi Trắc nghiệm). Here is a training example:

```
[
    {
        "question_id": "TN-102",
        "question_type": "Trắc nghiệm",
        "text": "Nam, nữ kết hôn với nhau phải từ đủ bao nhiêu tuổi trở lên?",
	"choices": {
		"A": "Nam từ đủ 20 tuổi trở lên, nữ từ đủ 18 tuổi trở lên.",
		"B": "Nam từ đủ 18 tuổi trở lên, nữ từ đủ 20 tuổi trở lên.",
		"C": "Nam từ đủ 21 tuổi trở lên, nữ từ đủ 19 tuổi trở lên.",
		"D": "Nam từ đủ 19 tuổi trở lên, nữ từ đủ 21 tuổi trở lên."
	},
        "relevant_articles": [
            {
                "law_id": "52/2014/QH13",
                "article_id": "8"
            }
        ],
	"answer": "A"
    }
]
```

For the multiple-choice questions, the answer must be "A", "B", "C" or "D".

3. Free-text questions (Câu hỏi tự luận). Here is a training example:

```
[
    {
        "question_id": "TL-103",
        "question_type": "Tự luận",
        "text": "Cơ quan nào có trách nhiệm thống nhất quản lý nhà nước về điện ảnh?",
        "relevant_articles": [
            {
                "law_id": "05/2022/QH15",
                "article_id": "45"
            }
        ],
	"answer": "Chính phủ"
    }
]
```
For the free-text questions, the answer is free-text and will be evaluated by human experts.


The principal evaluation measure is accuracy:

$$
\text{accuracy} = \frac{\text{the number of questions that were correctly answered}}{\text{the number of questions}}
$$

## Submission Details
Task 1: Legal Document Retrieval

The submission file for Task 1 should be in JSON format as follows:

```
[
    {
        "question_id": "TN-2",
        "relevant_articles": [
            {
                "law_id": "05/2022/QH15",
                "article_id": "95"
            }
        ]
    },
    ...
]
```

Task 2: Legal Question Answering

```
[
    {
        "question_id": "TL-3",
        "answer": <the answer>
    },
    ...
]
```

- Submission of Predictions: Participants must submit the files containing the systems' predictions for each task via email. For each task, participants are allowed to submit a maximum of 3 files, which should correspond to 3 different settings or methods for this task.

- Submission of Source Code: Participants are required to submit the source code of their method.

- Submission of Papers: Participants are required to submit a paper on their method and experimental results. Papers should conform to the standards set out on the KSE 2024 webpage (section Submission). At least one of the authors of an accepted paper has to present the paper at the ALQAC workshop of KSE 2024.