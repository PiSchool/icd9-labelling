#THe CLeaning Function
import string, re ,ftfy

def clean_note(x, removeNumbers = True, removeDoubleWords = True):
        '''
        This cleans the notes:
        1) uses translation to remove the punctuation, and replace it with nothing.
        2) the next part removes all new line carriages, tabs, and excess white space.
        3) returns cleaned string.

       
        '''
        x = ftfy.fix_text(x,remove_terminal_escapes=True, remove_control_chars=True, fix_line_breaks=True)
        #Convert to lower case 
        x = x.lower()
        translator = str.maketrans('', '', string.punctuation)
        x = x.translate(translator)

            #This removes words that have any numbers in it.
        if removeNumbers:
            x = re.sub(r'\w*\d\w*', '', x).strip()

        if removeDoubleWords:
            x = re.sub(r'\b(\w+)\s+\1\b', '', x).strip()

        x = re.sub( '\s+', ' ', x ).strip()


        return x

