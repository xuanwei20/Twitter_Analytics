import count_words

#mla.analyze_data(x_test=?)

dictionary = {1:"today is thursday",
              2: "tomorrow is friday",
              3: "when will it rain",
              4: "when can i see sunshine",
              5: "hurricane"}


if __name__ == '__main__':
    # One line invocation
    mla = count_words.MLAnalyzeData().analyze_mongo_data(dictionary)

    # Also possible to invoke the model once
    # mla = count_words.MLAnalyzeData()

    # And then call mla.analyze_mongo_data(dictionary) for every dictionary