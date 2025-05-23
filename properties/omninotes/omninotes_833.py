import sys
sys.path.append("..")
from kea import *

class Test(KeaTest):
    

    # @initialize()
    # def set_up(self):
    #     pass

    @mainPath()
    def change_language_to_chinese_mainpath(self):
        d(description="drawer open").click()
        d(text="SETTINGS").click()
        d(text="Interface").click()

    
    @precondition(lambda self: d(text="Interface").exists() and d(text="Language").exists())
    @rule()
    def change_language_to_chinese(self):
        
        d(text="Language").click()
        
        d(text="中文 (Chinese Simplified)").click()
        time.sleep(2)
        if d(text="OK").exists():
            d(text="OK").click()
            
        assert d(text="笔记").exists()



if __name__ == "__main__":
    t = Test()
    
    setting = Setting(
        apk_path="./apk/omninotes/OmniNotes-6.1.0beta2.apk",
        device_serial="emulator-5554",
        output_dir="../output/omninotes/833/guided",
        policy_name="guided"
    )
    start_kea(t,setting)
    
