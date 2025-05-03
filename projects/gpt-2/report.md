# Project: LLM Familiarization

The goal of this project is to develop basic familiarity with an LLM foundation model. Having just read the GPT-2 paper (Goodfellow et al. 2020) I chose the GPT-2 model.

# Observations:

The small model (`gpt2`) produces content that is borderline gibberish. It frequently repeats itself.

```
Input prompt: "Once upon a time"

gpt2: "Once upon a time, the world was a place of great beauty and great danger. The world was a place of great danger, and the world was a place of great danger. The world was a place of great danger, and the world was a place of great danger"
```

The 1.5B parameter model produces better results, but, it is nowhere near the performance of SOTA commercial products such as GPT-4o, etc.

```
Input prompt: Once upon a time

Output: Once upon a time, this meant we weren't playing the game.

The world's first major video game crash occurred in 1983, when more than two months of the original Donkey Kong were unaccounted for. When the game was finally found on a Nintendo 64 disc, it was unplayable.

For decades, video gaming was a niche hobby, especially in the early '90s, until Nintendo began to take its first serious steps toward broadening its appeal. The company's Virtual Console service, launched
```

## Quote Continuation

Prompting the `gpt2-xl` model with famous quotes often fails to yield results with any semblance of the expected continuation. I find this surprising considering the model is predicting the next word in the sequence.

```
Prompt: Tell me, friend, when did
Expected: Saruman the wise exchange reason for madness?
Actual: when did you last see a man who could make a woman laugh?"

Prompt: Many that live deserve death.
Expected: And some that die deserve life. Can you give it to them?
Actual: But I am not one of them.

Prompt: What does Marsellus Wallace look like?
Expected: Does he look like a ...
Actual: He's a tall, lean, athletic, and athletic-looking man.

Prompt: What is the chief end of man?
Expected: Man's chief end is to glorify God, and to enjoy Him for ever.
Actual: To know God, and to be happy.
```

## Quote Continuation w/ Contextual Prefix

When the prompt has a small contextual prefix we get better results, though still not the expected quote completion.

```
Prompt: Lord of the Rings quote: "Tell me, friend, when did
Actual: when did you last see a dragon?" ...

Prompt: Lord of the Rings quote: "Many that live deserve death.
Actual: But few that die deserve life." The quote is from the book of Ecclesiastes, and is attributed to the prophet Isaiah. ...

Prompt: Pulp Fiction Quote: "What does Marsellus Wallace look like?
Actual: "What does Marsellus Wallace look like? A fat, balding, middle-aged man with a mustache." Pulp Fiction Quote: "I'm not a murderer. I'm a murderer."...

Prompt: Westminister Shorter Catechism excerpt: "What is the chief end of man?
Actual: To know God, to love God, to serve God, to be obedient to God, to be holy, to be pure, to be happy, ...
```

Interestingly, the inclusion of the `"` in the prompt of the Westminister Shorter Catechism quote results in a dramatic difference:

```
Prompt: Westminister Shorter Catechism excerpt: What is the chief end of man?
Actual: The chief end of man is to glorify God.
```

### General Observations

- When the model is in `do_sample=False` we tend to get responses with the same phrase repeated over and over again. When `do_sample=True` the model can "break out" of whatever local optima its in and produce more varied responses.

## Extracting Information from News Articles

Prompting the model with a news article followed by a question yielded results that were often reasonable, if not confused on details (a result observed in Goodfellow 2020). As the authors noted, prompting the system with task context such as "TL;DR: " at the end of the article is effective at producing the expected task output (i.e., article summarization).

### Question: Argmax over semantic prefixes

- How can we quantize and optimize the prompt for the intended task domain? In other words, which variation of a the suffix prompt (e.g., "TL;DR: ") will best result in an article summary? Perhaps there is a way to search the embedding space for semantic signals, and then find find the argmax of such a signal?

## Ordered Steps

Below are some examples of prompting GPT-2 to provide a list of steps for achieving some task.

```
Prompt: Ordered steps to make a peanut butter and jelly sandwich:

Output:
1. Preheat oven to 350 degrees.

2. In a large bowl, combine peanut butter and jelly.

3. Add in flour, baking powder, baking soda, salt, and cinnamon.

4. Mix until well combined.

5. Add in chocolate chips and peanut butter mixture.

6. Mix until well combined.

7. Spread peanut butter and jelly mixture on a cookie sheet.

8. Bake for 10 minutes.
```

Observations:

- Perhaps this task is so simple that it is not well represented in the training corpus
- The LM clearly picks up on the theme of recipe instructions and retains the context of the sweet/savory nature of a PB&J (i.e., it didn't tell us to add meat or red wine).

```
Prompt: How to clear Chrome browser cache:

Output:
1. Open Chrome browser.

2. Click on the menu button (three horizontal lines) in the top right corner of the browser window.

3. Select Settings.

4. Select Advanced.

5. Select Clear browsing data.

6. Click on OK.

7. Close Chrome browser.

8. Restart your computer.

9. Open Chrome browser.
```

Observation:

- I suspect that this more technical task is better represented in the training corpus; i.e., those on the internet go to the internet to find help with internet related tasks.

```
Prompt: How to remain calm in if your parachute fails:

Output:
1. Don't panic.

2. Don't panic.

3. Don't panic.

4. Don't panic.

5. Don't panic.

6. Don't panic.

7. Don't panic.

8. Don't panic.

9. Don't panic.

10. Don't panic.

11. Don't panic.

12. Don't panic.

13. ...
```

- Notably, I didn't ask it "what to do", I asked it "how to remain calm".
- I wonder if the model is reflecting some dark humor in this response.
