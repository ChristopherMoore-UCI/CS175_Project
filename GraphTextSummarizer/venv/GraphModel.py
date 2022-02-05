import os
import spacy
import pytextrank
from icecream import ic

DEBUG = False


class Story:
    def __init__(self, text, highlights):
        self.text = text
        self.highlights = highlights


def get_text_and_highlights(file_name):
    try:
        with open(file_name) as openFile:
            text = []
            highlights = []
            at_highlights = False
            for line in openFile:
                if line != '\n':
                    if line.strip() == '@highlight' and at_highlights == False:
                        at_highlights = True
                    elif at_highlights == True and line.strip() != '@highlight':
                        highlights.append(line.strip())
                    elif at_highlights == False:
                        text.append(line.strip())
        return text, highlights
    except:
        return 'file: ' + file_name + ' not found'


def load_stories(prefix):
    stories = []
    for filename in os.listdir(prefix):
        text, highlights = get_text_and_highlights(prefix + '/' + filename)
        if DEBUG:
            print(filename)
            print('STORY')
            print(text)
            print('HIGHLIGHTS')
            print(highlights)
            print('\n')
        stories.append(clean_story(Story(text, highlights)))
    return stories


def clean_line(line):
    # get rid of non alpha numeric
    start_index = line.find('(CNN)')
    if start_index != -1:
        line = line[start_index + len('(CNN)'):]
    return line


def clean(text):
    cleaned_text = []
    for line in range(len(text)):
        cleaned_text.append(clean_line(text[line].strip()) if line == 0 else text[line].strip())
    return cleaned_text


def clean_story(story):
    story.text = clean(story.text)
    return story

def get_summary(text, nlp):
    doc = nlp('\n'.join(text))
    tr = doc._.textrank
    summary = []
    for sentence in tr.summary(limit_phrases=15, limit_sentences=min(4, len(doc) // 20)):
        summary.append(str(sentence).strip())
    return ' '.join(summary).replace('\n', ' ')


if __name__ == "__main__":

    nlp = spacy.load('en_core_web_sm')

    nlp.add_pipe('textrank')

    stories_location = ('data/cnn/stories')
    stories = load_stories(stories_location)

    for story in stories:
        print(story.highlights)
    generated_summaries = []
    story_highlights = []
    for i in range(len(stories)):
        print(i)
        # generated_summaries.append(get_summary(stories[i].text, nlp))
        story_highlights.append(stories[i].highlights)

    # create os directory for generated summaries
    generated_summaries_folder = 'generated_summaries'
    if not os.path.isdir(generated_summaries_folder):
        os.mkdir(generated_summaries_folder)

    # put generated summaries into generated summaries folder
    for i in range(len(generated_summaries)):
        print(i)
        output_file = open(generated_summaries_folder + '/summary' + str(i+1) + '.txt', 'w')
        output_file.write(generated_summaries[i])
        output_file.close()

    # create os directory for story highlights
    story_highlights_folder = 'story_highlights'
    if not os.path.isdir('story_highlights'):
        os.mkdir('story_highlights')

    # put story highlights into story highlights folder
    for i in range(len(story_highlights)):
        print(i)
        output_file = open(story_highlights_folder + '/summary' + str(i+1) + '.txt', 'w')
        output_file.write('. '.join(stories[i].highlights))
        output_file.close()
