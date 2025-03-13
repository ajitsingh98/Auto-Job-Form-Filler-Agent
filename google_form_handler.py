import json
import requests
import re
import pandas as pd

class GoogleFormHandler:
    ALL_DATA_FIELDS = "FB_PUBLIC_LOAD_DATA_"
    FORM_SESSION_TYPE_ID = 8
    ANY_TEXT_FIELD = "ANY TEXT!!"
    
    def __init__(self, url: str):
        self.url = url
        self.form_data = None
        self.entries = None
        self.page_count = 0
        
    def _get_form_response_url(self):
        ''' Convert form URL to form response URL '''
        url = self.url.replace('/viewform', '/formResponse')
        if not url.endswith('/formResponse'):
            if not url.endswith('/'):
                url += '/'
            url += 'formResponse'
        return url

    def _extract_script_variables(self, name: str, html: str):
        """ Extract a variable from a script tag in a HTML page """
        pattern = re.compile(r'var\s' + name + r'\s=\s(.*?);')
        match = pattern.search(html)
        if not match:
            return None
        value_str = match.group(1)
        return json.loads(value_str)

    def _get_fb_public_load_data(self):
        """ Get form data from a Google form URL """
        response = requests.get(self.url, timeout=10)
        if response.status_code != 200:
            print("Error! Can't get form data", response.status_code)
            return None
        return self._extract_script_variables(self.ALL_DATA_FIELDS, response.text)

    def _parse_entry(self, entry):
        entry_name = entry[1]
        entry_type_id = entry[3]
        result = []
        for sub_entry in entry[4]:
            info = {
                "id": sub_entry[0],
                "container_name": entry_name,
                "type": entry_type_id,
                "required": sub_entry[2] == 1,
                "name": ' - '.join(sub_entry[3]) if (len(sub_entry) > 3 and sub_entry[3]) else None,
                "options": [(x[0] or self.ANY_TEXT_FIELD) for x in sub_entry[1]] if sub_entry[1] else None,
            }
            result.append(info)
        return result

    def parse_form_entries(self):
        """ Parse the form entries and return a list of entries """
        url = self._get_form_response_url()
        self.form_data = self._get_fb_public_load_data()

        if not self.form_data or not self.form_data[1] or not self.form_data[1][1]:
            print("Error! Can't get form entries. Login may be required.")
            return None
        
        parsed_entries = []
        self.page_count = 0
        for entry in self.form_data[1][1]:
            if entry[3] == self.FORM_SESSION_TYPE_ID:
                self.page_count += 1
                continue
            parsed_entries += self._parse_entry(entry)

        # Collect email addresses if needed
        if self.form_data[1][10][6] > 1:
            parsed_entries.append({
                "id": "emailAddress",
                "container_name": "Email Address",
                "type": "required",
                "required": True,
                "options": "email address",
            })
        if self.page_count > 0:
            parsed_entries.append({
                "id": "pageHistory",
                "container_name": "Page History",
                "type": "required",
                "required": False,
                "options": "from 0 to (number of page - 1)",
                "default_value": ','.join(map(str, range(self.page_count + 1)))
            })
        
        self.entries = parsed_entries
        return parsed_entries

    def get_form_questions_df(self, only_required=False) -> pd.DataFrame:
        """
        Get form questions as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: Entry_ID, Question, Required, Field_Type, Selection_Type, Options
        """
        if not self.entries:
            self.parse_form_entries()
            
        if not self.entries:
            return pd.DataFrame()
            
        questions_data = []
        for entry in self.entries:
            question_text = entry['container_name']
            if entry.get('name'):
                question_text += f": {entry['name']}"
                
            # Determine selection type for multiple choice/checkbox fields
            selection_type = None
            if entry['type'] in [2, 3, 4]:  # Multiple choice, Dropdown, Checkboxes
                if entry['type'] == 2:
                    selection_type = "Single Choice"
                elif entry['type'] == 3:
                    selection_type = "Dropdown"
                elif entry['type'] == 4:
                    selection_type = "Multiple Choice"
                
            question_info = {
                'Entry_ID': f"entry.{entry['id']}" if entry.get('type') != "required" else entry['id'],
                'Question': question_text,
                'Required': entry['required'],
                'Field_Type': self.get_form_type_value_rule(entry['type']),
                'Selection_Type': selection_type,
                'Options': ', '.join(entry['options']) if entry.get('options') else None
            }
            questions_data.append(question_info)
        questions_df = pd.DataFrame(questions_data)
        questions_df = questions_df[questions_df["Entry_ID"]!="pageHistory"]
        if only_required:
            questions_df = questions_df[questions_df["Required"]==True]
        return questions_df

    def get_form_type_value_rule(self, type_id):
        ''' ------ TYPE ID ------ 
            0: Short answer
            1: Paragraph
            2: Multiple choice
            3: Dropdown
            4: Checkboxes
            5: Linear scale
            7: Grid choice
            9: Date
            10: Time
        '''
        type_mapping = {
            0: "Short answer",
            1: "Paragraph",
            2: "Multiple choice",
            3: "Dropdown",
            4: "Checkboxes",
            5: "Linear scale",
            7: "Grid choice",
            9: "Date",
            10: "Time",
            "required": "Required field"
        }
        return type_mapping.get(type_id, "any text")
    
    def fill_form_entries(self, fill_algorithm):
        """ Fill form entries with the provided fill_algorithm """
        for entry in self.entries:
            if entry.get('default_value'):
                continue
            # remove ANY_TEXT_FIELD from options to prevent choosing it
            options = (entry['options'] or [])[::]
            if self.ANY_TEXT_FIELD in options:
                options.remove(self.ANY_TEXT_FIELD)
            
            entry['default_value'] = fill_algorithm(entry['type'], entry['id'], options, 
                required=entry['required'], entry_name=entry['container_name'])
        return self.entries

    def get_form_submit_request(self, output="console", only_required=False, with_comment=True, fill_algorithm=None):
        ''' Get form request body data '''
        if not self.entries:
            self.parse_form_entries(only_required=only_required)

        if fill_algorithm:
            self.entries = self.fill_form_entries(fill_algorithm)
        if not self.entries:
            return None
        result = self.generate_form_request_dict(self.entries, with_comment)
        
        if output == "console":
            print(result)
        elif output == "return":
            return result
        else:
            # save as file
            with open(output, "w", encoding="utf-8") as f:
                f.write(result)
                print(f"Saved to {output}", flush=True)
            f.close()

    def submit_form(self, data):
        url = self._get_form_response_url()
        res = requests.post(url, data=data, timeout=5)
        if res.status_code == 200:
            print("Successfully submitted the form")
            return True
        else:
            print("Error! Can't submit form", res.status_code)
            return False


if __name__ == "__main__":
    # Example usage
    pass
