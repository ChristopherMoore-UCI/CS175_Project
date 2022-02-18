from rouge import Rouge
from collections import defaultdict


def get_text(file_path: str):
    """Extracts and returns the text from a file"""
    with open (file_path) as open_file:
        text = open_file.read()
    return text


def generate_rouge_scores(generated_summaries_dir: str, story_highlights_dir: str, summary_num: int, r: Rouge):
    """Generates and returns the rouge scores for given summary number
    'f' stands for f1_score, 'p' stands for precision, 'r' stands for recall."""
    generated_summary_content = get_text(generated_summaries_dir + '/summary' + str(summary_num) + '.txt')
    story_highlight_content = get_text(story_highlights_dir + '/summary' + str(summary_num) + '.txt')
    try:
        return r.get_scores(generated_summary_content, story_highlight_content)
    except ValueError as ve:
        return None


def calc_avg_rouge_scores(generated_summaries_dir: str, story_highlights_dir: str, starting_summary_num: int, \
                          ending_summary_num: int, r: Rouge):
    num_stories = ending_summary_num - starting_summary_num

    total_rouge_1 = defaultdict(int)
    total_rouge_2 = defaultdict(int)
    total_rouge_l = defaultdict(int)

    def add_to_dict(d1, d2):
        for key in d2.keys():
            d1[key] += d2[key]
        return d1

    def calc_avg(total_rouge: dict, story_count: int) -> dict:
        for key in total_rouge.keys():
            total_rouge[key] /= story_count
        return total_rouge

    for i in range(starting_summary_num, ending_summary_num):
        score = generate_rouge_scores(generated_summaries_dir, story_highlights_dir, i, r)
        if score:
            add_to_dict(total_rouge_1, score[0]['rouge-1'])
            add_to_dict(total_rouge_2, score[0]['rouge-2'])
            add_to_dict(total_rouge_l, score[0]['rouge-l'])
        else:
            num_stories -= 1
    rouge_1 = calc_avg(total_rouge_1, num_stories)
    rouge_2 = calc_avg(total_rouge_2, num_stories)
    rouge_l = calc_avg(total_rouge_l, num_stories)

    return rouge_1, rouge_2, rouge_l

# def print_result

if __name__ == '__main__':

    rouge = Rouge()

    scores = calc_avg_rouge_scores('generated_summaries', 'story_highlights', 1, 92580, rouge)
    for score in scores:
        print(score)


