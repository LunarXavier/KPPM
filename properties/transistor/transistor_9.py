import sys
sys.path.append("..")
from kea import *

class Test(KeaTest):

    @mainPath()
    def should_add_station_mainpath(self):
        d(resourceId="org.y20k.transistor:id/menu_add").click()
        d(className="android.widget.EditText").set_text("http://st01.dlf.de/dlf/01/128/mp3/stream.mp3")

    @precondition(
        lambda self: d(text="Add new station").exists() and 
        d(className="android.widget.EditText").get_text() != "Paste a valid streaming URL"     
    )
    @rule()
    def should_add_station(self):
        print(d(className="android.widget.EditText").get_text())
        d(text="ADD").click()
        time.sleep(3)
        if d(text="Download Issue").exists():
            print("Download Issue")
            return
        assert d(resourceId="org.y20k.transistor:id/list_item_textview").exists() 



if __name__ == "__main__":
    t = Test()
    
    setting = Setting(
        apk_path="./apk/transistor/1.1.4.apk",
        device_serial="emulator-5554",
        output_dir="../output/transistor/9/guided",
        policy_name="guided",
        number_of_events_that_restart_app = 100
    )
    start_kea(t,setting)
    
