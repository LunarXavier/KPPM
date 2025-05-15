import os
import logging
import random
import copy
import re
import time
from .utils import Time, generate_report, save_log, RULE_STATE
import networkx as nx
from abc import abstractmethod
from .input_event import (
    KEY_RotateDeviceToPortraitEvent,
    KEY_RotateDeviceToLandscapeEvent,
    KeyEvent,
    IntentEvent,
    ReInstallAppEvent,
    RotateDevice,
    RotateDeviceToPortraitEvent,
    RotateDeviceToLandscapeEvent,
    KillAppEvent,
    KillAndRestartAppEvent,
    SetTextEvent,
)
from .utg import UTG
from openai import OpenAI
# from .kea import utils
from .kea import CHECK_RESULT
from typing import TYPE_CHECKING, Dict

import sys
import io

# 修改标准输出的默认编码为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

if TYPE_CHECKING:
    from .input_manager import InputManager
    from .kea import Kea
    from .app import App
    from .device import Device

# Max number of restarts
MAX_NUM_RESTARTS = 5
# Max number of steps outside the app
MAX_NUM_STEPS_OUTSIDE = 10
MAX_NUM_STEPS_OUTSIDE_KILL = 10
# Max number of replay tries
MAX_REPLY_TRIES = 5
START_TO_GENERATE_EVENT_IN_POLICY = 2
# Max number of query llm
MAX_NUM_QUERY_LLM = 10

# Some input event flags
EVENT_FLAG_STARTED = "+started"
EVENT_FLAG_START_APP = "+start_app"
EVENT_FLAG_STOP_APP = "+stop_app"
EVENT_FLAG_EXPLORE = "+explore"
EVENT_FLAG_NAVIGATE = "+navigate"
EVENT_FLAG_TOUCH = "+touch"

# Policy taxanomy
POLICY_GUIDED = "guided"
POLICY_RANDOM = "random"
POLICY_NONE = "none"
POLICY_LLM = "llm"


class InputInterruptedException(Exception):
    pass


class InputPolicy(object):
    """
    This class is responsible for generating events to stimulate more app behaviour
    It should call AppEventManager.send_event method continuously
    """

    def __init__(self, device: "Device", app: "App", allow_to_generate_utg=True):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.time_recoder = Time()
        self.utg = UTG(device=device, app=app)
        self.device = device
        self.app = app
        self.event_count = 0

        self.last_event = None
        self.from_state = None
        self.to_state = None
        self.allow_to_generate_utg = allow_to_generate_utg
        self.triggered_bug_information = []
        self.time_needed_to_satisfy_precondition = []
        self.statistics_of_rules = {}

        self._num_restarts = 0
        self._num_steps_outside = 0
        self._event_trace = ""

    def start(self, input_manager: "InputManager"):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        # number of events that have been executed
        self.event_count = 0
        # self.input_manager = input_manager
        while input_manager.enabled and self.event_count < input_manager.event_count:
            try:
                # always try to close the keyboard on the device.
                # if self.device.is_harmonyos is False and hasattr(self.device, "u2"):
                #     self.device.u2.set_fastinput_ime(True)

                self.logger.info("Exploration event count: %d", self.event_count)

                if self.to_state is not None:
                    self.from_state = self.to_state
                else:
                    self.from_state = self.device.get_current_state()

                # set the from_state to droidbot to let the pdl get the state
                self.device.from_state = self.from_state

                if self.event_count == 0:
                    # If the application is running, close the application.
                    event = KillAppEvent(app=self.app)
                elif self.event_count == 1:
                    # start the application
                    event = IntentEvent(self.app.get_start_intent())
                else:
                    event = self.generate_event()

                if event is not None:
                    try:
                        self.device.save_screenshot_for_report(
                            event=event, current_state=self.from_state
                        )
                    except Exception as e:
                        self.logger.error("SaveScreenshotForReport failed: %s", e)
                        self.from_state = self.device.get_current_state()
                        self.device.save_screenshot_for_report(event=event, current_state=self.from_state)
                    input_manager.add_event(event)
                self.to_state = self.device.get_current_state()
                self.last_event = event
                if self.allow_to_generate_utg:
                    self.update_utg()

                bug_report_path = os.path.join(self.device.output_dir, "all_states")
                # TODO this function signature is too long?
                generate_report(
                    bug_report_path,
                    self.device.output_dir,
                    self.triggered_bug_information,
                    self.time_needed_to_satisfy_precondition,
                    self.device.cur_event_count,
                    self.time_recoder.get_time_duration(),
                    self.statistics_of_rules
                )

            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.info("stop sending events: %s" % e)
                self.logger.info("action count: %d" % self.event_count)
                break

            except RuntimeError as e:
                self.logger.info("RuntimeError: %s, stop sending events" % e)
                break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback

                traceback.print_exc()
            self.event_count += 1
        self.tear_down()

    def update_utg(self):
        self.utg.add_transition(self.last_event, self.from_state, self.to_state)

    def move_the_app_to_foreground_if_needed(self, current_state):
        """
        if the app is not running on the foreground of the device, then try to bring it back
        """
        if current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()

            # It seems the app stucks at some state, has been
            # 1) force stopped (START, STOP)
            #    just start the app again by increasing self.__num_restarts
            # 2) started at least once and cannot be started (START)
            #    pass to let viewclient deal with this case
            # 3) nothing
            #    a normal start. clear self.__num_restarts.

            if self._event_trace.endswith(
                    EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP
            ) or self._event_trace.endswith(EVENT_FLAG_START_APP):
                self._num_restarts += 1
                self.logger.info(
                    "The app had been restarted %d times.", self._num_restarts
                )
            else:
                self._num_restarts = 0

            # pass (START) through
            if not self._event_trace.endswith(EVENT_FLAG_START_APP):
                if self._num_restarts > MAX_NUM_RESTARTS:
                    # If the app had been restarted too many times, enter random mode
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                else:
                    # Start the app
                    self._event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    return IntentEvent(intent=start_app_intent)

        elif current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self._num_steps_outside += 1

            if self._num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self._num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self._event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                return go_back_event
        else:
            # If the app is in foreground
            self._num_steps_outside = 0

    @abstractmethod
    def tear_down(self):
        """ """
        pass

    @abstractmethod
    def generate_event(self):
        """
        generate an event
        @return:
        """
        pass

    @abstractmethod
    def generate_random_event_based_on_current_state(self):
        """
        generate an event
        @return:
        """
        pass


class KeaInputPolicy(InputPolicy):
    """
    state-based input policy
    """

    def __init__(self, device, app, kea: "Kea" = None, allow_to_generate_utg=True):
        super(KeaInputPolicy, self).__init__(device, app, allow_to_generate_utg)
        self.kea = kea
        # self.last_event = None
        # self.from_state = None
        # self.to_state = None

        # retrive all the rules from the provided properties
        for rule in self.kea.all_rules:
            self.statistics_of_rules[str(rule.function.__name__)] = {
                RULE_STATE.PRECONDITION_SATISFIED: 0,
                RULE_STATE.PROPERTY_CHECKED: 0,
                RULE_STATE.POSTCONDITION_VIOLATED: 0,
                RULE_STATE.UI_OBJECT_NOT_FOUND: 0
            }

    def run_initializer(self):
        if self.kea.initializer is None:
            self.logger.warning("No initializer")
            return

        result = self.kea.execute_initializer(self.kea.initializer)
        if (
                result == CHECK_RESULT.PASS
        ):  # why only check `result`, `result` could have different values.
            self.logger.info("-------initialize successfully-----------")
        else:
            self.logger.error("-------initialize failed-----------")

    def check_rule_whose_precondition_are_satisfied(self):
        """
        TODO should split the function
        #! xixian - agree to split the function
        """
        # ! TODO - xixian - should we emphasize the following data structure is a dict?
        rules_ready_to_be_checked = (
            self.kea.get_rules_whose_preconditions_are_satisfied()
        )
        rules_ready_to_be_checked.update(self.kea.get_rules_without_preconditions())
        if len(rules_ready_to_be_checked) == 0:
            self.logger.debug("No rules match the precondition")
            return

        candidate_rules_list = list(rules_ready_to_be_checked.keys())
        # randomly select a rule to check
        rule_to_check = random.choice(candidate_rules_list)

        if rule_to_check is not None:
            self.logger.info(f"-------Check Property : {rule_to_check}------")
            self.statistics_of_rules[str(rule_to_check.function.__name__)][
                RULE_STATE.PROPERTY_CHECKED
            ] += 1
            precondition_page_index = self.device.cur_event_count
            # check rule, record relavant info and output log
            result = self.kea.execute_rule(
                rule=rule_to_check, keaTest=rules_ready_to_be_checked[rule_to_check]
            )
            if result == CHECK_RESULT.ASSERTION_FAILURE:
                self.logger.error(
                    f"-------Postcondition failed. Assertion error, Property:{rule_to_check}------"
                )
                self.logger.debug(
                    "-------time from start : %s-----------"
                    % str(self.time_recoder.get_time_duration())
                )
                self.statistics_of_rules[str(rule_to_check.function.__name__)][
                    RULE_STATE.POSTCONDITION_VIOLATED
                ] += 1
                postcondition_page__index = self.device.cur_event_count
                self.triggered_bug_information.append(
                    (
                        (precondition_page_index, postcondition_page__index),
                        self.time_recoder.get_time_duration(),
                        rule_to_check.function.__name__,
                    )
                )
            elif result == CHECK_RESULT.PASS:
                self.logger.info(
                    f"-------Post condition satisfied. Property:{rule_to_check} pass------"
                )
                self.logger.debug(
                    "-------time from start : %s-----------"
                    % str(self.time_recoder.get_time_duration())
                )

            elif result == CHECK_RESULT.UI_NOT_FOUND:
                self.logger.error(
                    f"-------Execution failed: UiObjectNotFound during exectution. Property:{rule_to_check}-----------"
                )
                self.statistics_of_rules[str(rule_to_check.function.__name__)][
                    RULE_STATE.UI_OBJECT_NOT_FOUND
                ] += 1
            elif result == CHECK_RESULT.PRECON_NOT_SATISFIED:
                self.logger.info("-------Precondition not satisfied-----------")
            else:
                raise AttributeError(f"Invalid property checking result {result}")

    def generate_event(self):
        """
        generate an event
        @return:
        """
        pass

    def update_utg(self):
        self.utg.add_transition(self.last_event, self.from_state, self.to_state)


class RandomPolicy(KeaInputPolicy):
    """
    generate random event based on current app state
    """

    def __init__(
            self,
            device,
            app,
            kea=None,
            restart_app_after_check_property=False,
            number_of_events_that_restart_app=100,
            clear_and_reinstall_app=False,
            allow_to_generate_utg=True,
            disable_rotate=False,
            output_dir=None
    ):
        super(RandomPolicy, self).__init__(device, app, kea, allow_to_generate_utg)
        self.restart_app_after_check_property = restart_app_after_check_property
        self.number_of_events_that_restart_app = number_of_events_that_restart_app
        self.clear_and_reinstall_app = clear_and_reinstall_app
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        save_log(self.logger, self.output_dir)
        self.disable_rotate = disable_rotate
        self.last_rotate_events = KEY_RotateDeviceToPortraitEvent

    def generate_event(self):
        """
        generate an event
        @return:
        """

        if self.event_count == START_TO_GENERATE_EVENT_IN_POLICY or isinstance(
                self.last_event, ReInstallAppEvent
        ):
            self.run_initializer()
            self.from_state = self.device.get_current_state()
        current_state = self.from_state
        if current_state is None:
            time.sleep(5)
            return KeyEvent(name="BACK")

        if self.event_count % self.number_of_events_that_restart_app == 0:
            if self.clear_and_reinstall_app:
                self.logger.info(
                    "clear and reinstall app after %s events"
                    % self.number_of_events_that_restart_app
                )
                return ReInstallAppEvent(self.app)
            self.logger.info(
                "restart app after %s events" % self.number_of_events_that_restart_app
            )
            return KillAndRestartAppEvent(app=self.app)

        rules_to_check = self.kea.get_rules_whose_preconditions_are_satisfied()
        for rule_to_check in rules_to_check:
            self.statistics_of_rules[str(rule_to_check.function.__name__)][
                RULE_STATE.PRECONDITION_SATISFIED
            ] += 1

        if len(rules_to_check) > 0:
            t = self.time_recoder.get_time_duration()
            self.time_needed_to_satisfy_precondition.append(t)
            self.logger.debug(
                "has rule that matches the precondition and the time duration is "
                + t
            )
            if random.random() < 0.5:
                self.logger.info("Check property")
                self.check_rule_whose_precondition_are_satisfied()
                if self.restart_app_after_check_property:
                    self.logger.debug("restart app after check property")
                    return KillAppEvent(app=self.app)
                return None
            else:
                self.logger.info("Don't check the property due to the randomness")

        event = self.generate_random_event_based_on_current_state()

        return event

    def generate_random_event_based_on_current_state(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        current_state = self.from_state
        self.logger.debug("Current state: %s" % current_state.state_str)
        event = self.move_the_app_to_foreground_if_needed(current_state)
        if event is not None:
            return event

        possible_events = current_state.get_possible_input()
        possible_events.append(KeyEvent(name="BACK"))
        if not self.disable_rotate:
            possible_events.append(RotateDevice())

        self._event_trace += EVENT_FLAG_EXPLORE

        event = random.choice(possible_events)
        if isinstance(event, RotateDevice):
            # select a rotate event with different direction than last time
            if self.last_rotate_events == KEY_RotateDeviceToPortraitEvent:
                self.last_rotate_events = KEY_RotateDeviceToLandscapeEvent
                event = (
                    RotateDeviceToLandscapeEvent()
                )
            else:
                self.last_rotate_events = KEY_RotateDeviceToPortraitEvent
                event = RotateDeviceToPortraitEvent()
        return event


class GuidedPolicy(KeaInputPolicy):
    """
    generate events around the main path
    """

    def __init__(self, device, app, kea=None, allow_to_generate_utg=True, disable_rotate=False, output_dir=None):
        super(GuidedPolicy, self).__init__(device, app, kea, allow_to_generate_utg)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        save_log(self.logger, self.output_dir)
        self.disable_rotate = disable_rotate
        if len(self.kea.all_mainPaths):
            self.logger.info("Found %d mainPaths" % len(self.kea.all_mainPaths))
        else:
            self.logger.error("No mainPath found")

        self.main_path = None
        self.execute_main_path = True

        self.current_index_on_main_path = 0
        self.max_number_of_mutate_steps_on_single_node = 20
        self.current_number_of_mutate_steps_on_single_node = 0
        self.number_of_events_that_try_to_find_event_on_main_path = 0
        self.index_on_main_path_after_mutation = -1
        self.mutate_node_index_on_main_path = 0

        self.last_random_text = None
        self.last_rotate_events = KEY_RotateDeviceToPortraitEvent

    def select_main_path(self):
        if len(self.kea.all_mainPaths) == 0:
            self.logger.error("No mainPath")
            return
        self.main_path = random.choice(self.kea.all_mainPaths)
        # self.path_func, self.main_path =  self.kea.parse_mainPath(self.main_path)
        self.path_func, self.main_path = self.main_path.function, self.main_path.path
        self.logger.info(
            f"Select the {len(self.main_path)} steps mainPath function: {self.path_func}"
        )
        self.main_path_list = copy.deepcopy(self.main_path)
        self.max_number_of_events_that_try_to_find_event_on_main_path = min(
            10, len(self.main_path)
        )
        self.mutate_node_index_on_main_path = len(self.main_path)

    def generate_event(self):
        """ """
        current_state = self.from_state

        # Return relevant events based on whether the application is in the foreground.
        event = self.move_the_app_to_foreground_if_needed(current_state)
        if event is not None:
            return event

        if ((self.event_count == START_TO_GENERATE_EVENT_IN_POLICY)
                or isinstance(self.last_event, ReInstallAppEvent)):
            self.select_main_path()
            self.run_initializer()
            time.sleep(2)
            self.from_state = self.device.get_current_state()
        if self.execute_main_path:
            event_str = self.get_next_event_from_main_path()
            if event_str:
                self.logger.info("*****main path running*****")
                self.kea.execute_event_from_main_path(event_str)
                return None
        if event is None:
            # generate event aroud the state on the main path
            event = self.mutate_the_main_path()

        return event

    def stop_mutation(self):
        self.index_on_main_path_after_mutation = -1
        self.number_of_events_that_try_to_find_event_on_main_path = 0
        self.execute_main_path = True
        self.current_number_of_mutate_steps_on_single_node = 0
        self.current_index_on_main_path = 0
        self.mutate_node_index_on_main_path -= 1
        if self.mutate_node_index_on_main_path == -1:
            self.mutate_node_index_on_main_path = len(self.main_path)
            return ReInstallAppEvent(app=self.app)
        self.logger.info(
            "reach the max number of mutate steps on single node, restart the app"
        )
        return KillAndRestartAppEvent(app=self.app)

    def mutate_the_main_path(self):
        event = None
        self.current_number_of_mutate_steps_on_single_node += 1

        if (
                self.current_number_of_mutate_steps_on_single_node
                >= self.max_number_of_mutate_steps_on_single_node
        ):
            # try to find an event from the main path that can be executed on current state
            if (
                    self.number_of_events_that_try_to_find_event_on_main_path
                    <= self.max_number_of_events_that_try_to_find_event_on_main_path
            ):
                self.number_of_events_that_try_to_find_event_on_main_path += 1
                # if reach the state that satsfies the precondition, check the rule and turn to execute the main path.
                if self.index_on_main_path_after_mutation == len(self.main_path_list):
                    self.logger.info(
                        "reach the end of the main path that could satisfy the precondition"
                    )
                    rules_to_check = self.kea.get_rules_whose_preconditions_are_satisfied()
                    for rule_to_check in rules_to_check:
                        self.statistics_of_rules[str(rule_to_check.function.__name__)][
                            RULE_STATE.PRECONDITION_SATISFIED
                        ] += 1
                    if len(rules_to_check) > 0:
                        t = self.time_recoder.get_time_duration()
                        self.time_needed_to_satisfy_precondition.append(t)
                        self.logger.debug(
                            "has rule that matches the precondition and the time duration is "
                            + t
                        )
                        self.logger.info("Check property")
                        self.check_rule_whose_precondition_are_satisfied()
                    return self.stop_mutation()

                # find if there is any event in the main path that could be executed on currenty state
                event_str = self.get_event_from_main_path()
                try:
                    self.kea.execute_event_from_main_path(event_str)
                    self.logger.info("find the event in the main path")
                    return None
                except Exception:
                    self.logger.info("can't find the event in the main path")
                    return self.stop_mutation()

            return self.stop_mutation()

        self.index_on_main_path_after_mutation = -1

        if len(self.kea.get_rules_whose_preconditions_are_satisfied()) > 0:
            # if the property has been checked, don't return any event
            t = self.time_recoder.get_time_duration()
            self.time_needed_to_satisfy_precondition.append(t)
            self.logger.debug(
                "has rule that matches the precondition and the time duration is "
                + t
            )
            if random.random() < 0.5:
                self.logger.info("Check property")
                self.check_rule_whose_precondition_are_satisfied()
                return None
            else:
                self.logger.info("Don't check the property due to the randomness")

        event = self.generate_random_event_based_on_current_state()
        return event

    def get_next_event_from_main_path(self):
        """
        get a next event when execute on the main path
        """
        if self.current_index_on_main_path == self.mutate_node_index_on_main_path:
            self.logger.info(
                "reach the mutate index, start mutate on the node %d"
                % self.mutate_node_index_on_main_path
            )
            self.execute_main_path = False
            return None

        self.logger.info(
            "execute node index on main path: %d" % self.current_index_on_main_path
        )
        u2_event_str = self.main_path_list[self.current_index_on_main_path]
        if u2_event_str is None:
            self.logger.warning(
                "event is None on main path node %d" % self.current_index_on_main_path
            )
            self.current_index_on_main_path += 1
            return self.get_next_event_from_main_path()
        self.current_index_on_main_path += 1
        return u2_event_str

    def get_ui_element_dict(self, ui_element_str: str) -> Dict[str, str]:
        """
        get ui elements of the event
        """
        start_index = ui_element_str.find("(") + 1
        end_index = ui_element_str.find(")", start_index)

        if start_index != -1 and end_index != -1:
            ui_element_str = ui_element_str[start_index:end_index]
        ui_elements = ui_element_str.split(",")

        ui_elements_dict = {}
        for ui_element in ui_elements:
            attribute_name, attribute_value = ui_element.split("=")
            attribute_name = attribute_name.strip()
            attribute_value = attribute_value.strip()
            attribute_value = attribute_value.strip('"')
            ui_elements_dict[attribute_name] = attribute_value
        return ui_elements_dict

    def get_event_from_main_path(self):
        """
        get an event can lead current state to go back to the main path
        """
        if self.index_on_main_path_after_mutation == -1:
            for i in reversed(range(len(self.main_path_list))):
                event_str = self.main_path_list[i]
                ui_elements_dict = self.get_ui_element_dict(event_str)
                current_state = self.from_state
                view = current_state.get_view_by_attribute(ui_elements_dict)
                if view is None:
                    continue
                self.index_on_main_path_after_mutation = i + 1
                return event_str
        else:
            event_str = self.main_path_list[self.index_on_main_path_after_mutation]
            ui_elements_dict = self.get_ui_element_dict(event_str)
            current_state = self.from_state
            view = current_state.get_view_by_attribute(ui_elements_dict)
            if view is None:
                return None
            self.index_on_main_path_after_mutation += 1
            return event_str
        return None

    def generate_random_event_based_on_current_state(self):
        """
        generate an event based on current UTG to explore the app
        @return: InputEvent
        """
        current_state = self.from_state
        self.logger.info("Current state: %s" % current_state.state_str)
        event = self.move_the_app_to_foreground_if_needed(current_state)
        if event is not None:
            return event

        # Get all possible input events
        possible_events = current_state.get_possible_input()

        # if self.random_input:
        #     random.shuffle(possible_events)
        possible_events.append(KeyEvent(name="BACK"))
        if not self.disable_rotate:
            possible_events.append(RotateDevice())

        self._event_trace += EVENT_FLAG_EXPLORE

        event = random.choice(possible_events)
        if isinstance(event, RotateDevice):
            if self.last_rotate_events == KEY_RotateDeviceToPortraitEvent:
                self.last_rotate_events = KEY_RotateDeviceToLandscapeEvent
                event = RotateDeviceToLandscapeEvent()
            else:
                self.last_rotate_events = KEY_RotateDeviceToPortraitEvent
                event = RotateDeviceToPortraitEvent()

        return event


class LLMPolicy(RandomPolicy):
    """
    use LLM to generate input when detected ui tarpit
    """
    def __init__(
            self,
            device,
            app,
            kea=None,
            restart_app_after_check_property=False,
            number_of_events_that_restart_app=100,
            clear_and_restart_app_data_after_100_events=False,
            allow_to_generate_utg=False,
            output_dir=None
    ):
        super(LLMPolicy, self).__init__(device, app, kea, output_dir=output_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)  # 设置logger级别为DEBUG
        # 如果没有handler，添加一个
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)  # 设置handler级别为DEBUG
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.output_dir = output_dir
        save_log(self.logger, self.output_dir)
        # 历史动作列表
        self.__action_history = []
        # 历史页面描述列表
        self.__activity_history = []
        # 记录事件生成来源，1表示使用LLM生成事件，0表示随机生成事件
        self.record_llm_execution = []
        self.from_state = None

    def start(self, input_manager: "InputManager"):
        """
        LLM引导策略入口
        """
        self.event_count = 0
        self.input_manager = input_manager
        # 事件计数器 < 输入管理器事件计数
        while input_manager.enabled and self.event_count < input_manager.event_count:
            try:
                if self.device.is_harmonyos == False and hasattr(self.device, "u2"):
                    self.device.u2.set_fastinput_ime(True)

                self.logger.info("Exploration action count: %d" % self.event_count)

                if self.to_state is not None:
                    self.from_state = self.to_state
                else:
                    self.from_state = self.device.get_current_state()

                if self.event_count == 0:
                    # If the application is running, close the application.
                    event = KillAppEvent(app=self.app)
                elif self.event_count == 1:
                    # 打开app并执行初始化事件流
                    event = IntentEvent(self.app.get_start_intent())
                else:
                    # 检测是否陷入局部探索
                    if len(self.__activity_history) == 0:
                        is_trapped = False
                    else:
                        is_trapped = self.detect_local_exploration(self.__action_history, self.__activity_history)
                    is_stuck = False
                    # 如果大模型引导10步仍然无法离开局部探索，则认为无法逃脱
                    if len(self.record_llm_execution) >= 10 and all(x == 1 for x in self.record_llm_execution[-10:]):
                        is_stuck = True
                    # 如果无法逃脱，则重启应用
                    if is_trapped == True and is_stuck == True:
                        self.logger.info("LLM couldn't find an escape route. Restarting app.")
                        event = KillAndRestartAppEvent(app=self.app)
                    # 如果陷入局部探索，由大模型引导离开
                    elif is_trapped == True:
                        event = self.generate_llm_event()
                        self.record_llm_execution.append(1)
                    # 否则进行随机事件生成
                    else:
                        event = self.generate_random_event()
                        self.record_llm_execution.append(0)
                
                # 保存屏幕截图
                if event is not None:
                    try:
                        # 动作执行在原状态下
                        self.device.save_screenshot_for_report(event=event, current_state=self.device.get_current_state())
                    except Exception as e:
                        self.logger.error("SaveScreenshotForReport failed: %s", e)
                        self.from_state = self.device.get_current_state()
                        self.device.save_screenshot_for_report(event=event, current_state=self.from_state)
                    # 执行动作
                    input_manager.add_event(event)
                
                # state是DeviceState对象，获取事件执行后的状态
                self.to_state = self.device.get_current_state()
                self.last_event = event
                # 如果重启应用，则清空历史记录
                if isinstance(event, KillAndRestartAppEvent) or isinstance(event, ReInstallAppEvent) or isinstance(event, KillAppEvent):
                    self.__action_history = []
                    self.__activity_history = []
                    self.record_llm_execution = []
                # 将当前执行的事件和当前状态记录在历史记录中
                elif not isinstance(event, IntentEvent):
                    self.__action_history.append(self.to_state.get_event_desc(event))
                    self.__activity_history.append(self.get_state_description(self.to_state))
                # 更新utg图
                if self.allow_to_generate_utg:
                    self.update_utg()
                # 生成bug_report
                bug_report_path = os.path.join(self.device.output_dir, "all_states")
                generate_report(
                    bug_report_path,
                    self.device.output_dir,
                    self.triggered_bug_information,
                    self.time_needed_to_satisfy_precondition,
                    self.device.cur_event_count,
                    self.time_recoder.get_time_duration(),
                )
            except KeyboardInterrupt:
                break
            except InputInterruptedException as e:
                self.logger.info("stop sending events: %s" % e)
                self.logger.info("action count: %d" % self.event_count)
                break
            except RuntimeError as e:
                self.logger.info("RuntimeError: %s, stop sending events" % e)
                break
            except Exception as e:
                self.logger.warning("exception during sending events: %s" % e)
                import traceback
                traceback.print_exc()
            self.event_count += 1
        # 结束
        self.tear_down()
        
    def check_property(self):
        """
        判断是否重启应用，是否满足前置条件
        """
        # 如果达到预定的次数则重启应用
        if self.event_count % self.number_of_events_that_restart_app == 0:
            if self.clear_and_reinstall_app:
                self.logger.info("Clear and reinstall app after %s events." % self.number_of_events_that_restart_app)
                return ReInstallAppEvent(self.app)
            self.logger.info("Restart app after %s events." % self.number_of_events_that_restart_app)
            return KillAndRestartAppEvent(app=self.app)
        
        # 判断是否满足前置条件
        rules_to_check = self.kea.get_rules_whose_preconditions_are_satisfied()
        for rule_to_check in rules_to_check:
            self.statistics_of_rules[str(rule_to_check.function.__name__)][RULE_STATE.PRECONDITION_SATISFIED] += 1
        # 如果满足前置条件，则以一半的概率执行测试
        if len(rules_to_check) > 0:
            t = self.time_recoder.get_time_duration()
            self.time_needed_to_satisfy_precondition.append(t)
            self.logger.debug("has rule that matches the precondition and the time duration is {}.".format(t))
            if random.random() < 0.5:
                self.logger.info("Check property")
                self.check_rule_whose_precondition_are_satisfied()
                if self.restart_app_after_check_property:
                    self.logger.info("Restart app after check property.")
                    return KillAppEvent(app=self.app)
                return None
            else:
                self.logger.info("Don't check the property due to the randomness.")
        return None

    def generate_random_event(self):
        """
        随机生成事件，由大模型判断是否会触发焦油坑，如果会则重新生成
        """
        # 应用刚安装时，执行初始化应用，并记录当前页面语义
        if self.event_count == START_TO_GENERATE_EVENT_IN_POLICY or isinstance(self.last_event, ReInstallAppEvent):
            self.run_initializer()
            self.from_state = self.device.get_current_state()
            self.__action_history.append("initialize the application")
            self.__activity_history.append(self.get_state_description(self.from_state))

        current_state = self.from_state
        
        if current_state is None:
            time.sleep(5)
            return KeyEvent(name="BACK")
        
        if (len(self.__activity_history) == 0):
            self.__action_history.append("kill and restart the application")
            self.__activity_history.append(self.get_state_description(current_state))
        
        # 检查是否满足前置条件
        result = self.check_property()
        if result is not None:
            return result
        
        # 避免UI陷阱
        banned_list = set()
        while True:
            event = self.generate_random_event_based_on_current_state()
            # 如果是Intent操作或者是Rotate操作，则直接执行
            if isinstance(event, IntentEvent) or isinstance(event, RotateDevice):
                break
            event_desc = current_state.get_event_desc(event)

            if event_desc in banned_list:
                continue
                
            if self.is_risky_event(current_state, event):
                # self.logger.info(f"LLM判断该事件具有高风险：{event_desc}，加入ban列表。")
                banned_list.add(event_desc)
            else:
                # 找到合法事件
                break  
        return event
    
    def _query_llm(self, prompt, model_name="deepseek-chat"):
        """
        对话LLM
        """
        deepseek_url = "https://api.deepseek.com"  # DeepSeek 的 API 地址
        deepseek_key = ""  # 替换为你的 DeepSeek API 密钥
        client = OpenAI(api_key=deepseek_key, base_url=deepseek_url)

        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            messages=messages, model=model_name, timeout=30, stream=False
        )
        res = completion.choices[0].message.content
        return res

    def get_state_description(self, state):
        """
        获取状态的自然语言描述
        """
        prompt = (
            f"Below is a structured description of a page. Your task is to describe the content and features of this page using a simple sentence.\n"
            f"\"\"\"\n{state.get_semantic_info_tree()}\n\"\"\"\n\n"
            f"This is an example of the output content:\n"
            f"\"\"\"\nA note editing screen featuring a title input field, content editor, toolbar (with attachment, category, tag, and share options), and a reminder setting.\n\"\"\"\n\n"
            f"Imitate the example and output the page description. No other content is allowed.\n"
        )
        response = self._query_llm(prompt).strip()
        self.logger.info(f"State description: {response}")
        return response
    
    def generate_random_event_based_on_current_state(self):
        """
        基于当前状态随机生成事件
        """
        current_state = self.from_state
        # 如果app处于后台，则将app放置在前台
        event = self.move_the_app_to_foreground_if_needed(current_state)
        if event is not None:
            return event
        # 获取所有可选事件  
        possible_events = current_state.get_possible_input()
        possible_events.append(KeyEvent(name="BACK"))
        if not self.disable_rotate:
            possible_events.append(RotateDevice())

        self._event_trace += EVENT_FLAG_EXPLORE
        # 随机选择一个事件
        event = random.choice(possible_events)
        # 处理旋转操作，确保与先前的旋转操作方向不同
        if isinstance(event, RotateDevice):
            # select a rotate event with different direction than last time
            if self.last_rotate_events == KEY_RotateDeviceToPortraitEvent:
                self.last_rotate_events = KEY_RotateDeviceToLandscapeEvent
                event = RotateDeviceToLandscapeEvent()
            else:
                self.last_rotate_events = KEY_RotateDeviceToPortraitEvent
                event = RotateDeviceToPortraitEvent()
        return event

    def is_risky_event(self, current_state, event):
        """
        判断事件是否具有高风险
        """
        event_desc = current_state.get_event_desc(event)
        prompt = (
            f"An application is being explored, and an action will be performed on a page to navigate to a new page.\n"
            f"However, this action may cause the exploration to stall or result in leaving the application.\n"
            f"Here are a few examples:\n\"\"\"\n"
            f"1. Clicking the \"Log Out\" option on the \"Account\" page leave you stuck on the login page beacause you don't konw the account and secret key.\n"
            f"2. Clicking a hyperlink within the page cause you to leave the application.\n"
            f"3. Go back on the main page cause you to leave the application.\n\"\"\"\n"
            f"Note that go back to another page will not result in leaving the application.\n\n"
            f"Your task is to identify such situations.\n"
            f"The description of the current page is:\n\"\"\"\n{self.__activity_history[-1]}\n\"\"\"\n\n"
            f"The next action to be taken on this page will be: \n\"\"\"\n{event_desc}.\n\"\"\"\n\n"
            f"Determine whether this action will cause the exploration to stall or leave the application.\n"
            f"Give your answer and reason. The answer must be placed between labels <answer> </answer> and it must be either YES or NO. The reason must be placed between labels <reason> </reason>.\n"
            f"This is an example of output content:\n"
            f"\"\"\"\n<answer>\nYES\n<\answer>\n<reason>\nSince the current page is the main page, going back will result in leaving the application.\n<\reason>\n\"\"\"\n\n"
            f"Imitate the example and output the answer and reason. No other content is allowed.\n"
        )
        self.logger.info("Judging risky event...")
        # self.logger.info(f'[is_risky_event] LLM prompt:\n{prompt}')
        response = self._query_llm(prompt).strip()
        # self.logger.info(f'[is_risky_event] LLM response:\n{response}')
        
        # 提取answer标签内容
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        answer = answer_match.group(1).replace('\n', '') if answer_match else ""
        # 提取reason标签内容
        reason_pattern = r'<reason>(.*?)</reason>'
        reason_match = re.search(reason_pattern, response, re.DOTALL)
        reason = reason_match.group(1).replace('\n', '') if reason_match else ""
        
        if answer == "" or reason == "":
            self.logger.warning("[is_risky_event] Invalid response from LLM. Unable to parse the answer and reason.")
            self.logger.warning(f"LLM response:\n{response}")
        if answer == "YES":
            self.logger.warning(f"Find a risky event!\nThe event is {event_desc}.\nThe reason is: {reason}\n")
            return True
        return False
    
    def detect_local_exploration(self, action_history, activity_history):
        """
        大模型判断是否处于局部探索
        """
        # 维护一个滑动窗口，记录最近访问的20个界面
        action_window = action_history[-20:]
        activity_window = activity_history[-20:]

        task_prompt = (
                f"An application is being explored, and an action will be performed on a page to navigate to a new page.\n"
                f"However, the current exploration may have become stuck in local exploration.\n"
                f"Here are two conditions of local exploration:\n\"\"\"\n"
                f"1. Stay on the same page for more than 3 steps.\n"
                f"2. After performing a certain action, the user navigates from Page A to Page B, and after executing another action, returns to Page A. This cycle is repeated more than 3 times.\n\"\"\"\n\n"
                f"Your task is to determine whether the current exploration has fallen into local exploration.\n"
        )
        
        if len(action_window) == 0:
            info_prompt = "Currently, no exploration has been conducted.\n"
        else:
            info_prompt = (
                f"The performed actions and corresponding pages explored so far are:\n\"\"\"\n" + \
                "\n".join(f"<{str(action)}> -> <{str(activity)}>" for action, activity in zip(action_window, activity_window)) + \
                "\n\"\"\"\n\n"
            )

        output_prompt = (
            f"Determine whether the current exploration has become a case of local exploration.\n"
            f"Give your answer and reason. The answer must be placed between labels <answer> </answer> and it must be either YES or NO. The reason must be placed between labels <reason> </reason>. "
            f"If the answer is YES, give the description of the local exploration. The description must be placed between labels <description></description>. "
            f"The description should include all the actions that have been performed to leave the local exploration instead of omitting some of them.\n"
            f"This is an example of the output content with the \'NO\' answer:\n\"\"\"\n"
            f"<answer>\nNO\n</answer>\n"
            f"<reason>\nAccording to the performed actions and corresponding pages explored so far, "
            f"the app was navigated to different pages and did not get stuck in a loop, so neither of the above two cases was satisfied.\n</reason>\n\"\"\"\n\n"
            f"This is an example of the output content with the \'YES\' answer:\n\"\"\"\n"
            f"<answer>\nYES\n</answer>\n"
            f"<reason>\nThe exploration has performed more than 3 actions while remaining on the same page (a password setup screen), satisfying condition 1 of local exploration."
            f"Additionally, many actions are repetitive (e.g., clicking \"REMOVE PASSWORD\" or \"PASSWORD FORGOTTEN\" multiple times without navigating away), indicating potential local exploration.\n</reason>\n"
            f"<description>\nThe current exploration is stuck on a password setup screen. "
            f"There are many actions are performed on the page, including clicking \"REMOVE PASSWORD\" button, clicking \"PASSWORD FORGOTTEN\", scrolling and entering password. "
            f"All of the actions fail to leave this page.\n</description>\n\"\"\"\n\n"
            f"Imitate the example and output the answer and reason. No other content is allowed.\n"
        )

        # 判断是否处于局部探索
        prompt = f"{task_prompt}{info_prompt}{output_prompt}"
        self.logger.info("Detecting local exploration...")
        # self.logger.info(f'[detect_local_exploration] LLM prompt:\n{prompt}')
        
        response = self._query_llm(prompt).strip()
        # self.logger.info(f'[detect_local_exploration] LLM response:\n{response}')
        
        # 提取answer标签内容
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        answer = answer_match.group(1).replace('\n', '') if answer_match else ""
        # 提取reason标签内容
        reason_pattern = r'<reason>(.*?)</reason>'
        reason_match = re.search(reason_pattern, response, re.DOTALL)
        reason = reason_match.group(1).replace('\n', '') if reason_match else ""
        
        if answer == "" or reason == "":
            self.logger.warning("[detect_local_exploration] Invalid response from LLM. Unable to parse the answer and reason.")
            self.logger.warning(f"LLM response:\n{response}")
            
        if answer == "YES":
            self.logger.warning(f"Detect local exploration!\nThe reason is {reason}\n")
            description_pattern = r'<description>(.*?)</description>'
            description_match = re.search(description_pattern, response, re.DOTALL)
            description = description_match.group(1).replace('\n', '') if description_match else ""
            self.description = description
            return True
        return False

    def generate_llm_event(self):
        """
        大模型生成事件
        """
        # 检查是否满足前置条件
        result = self.check_property()
        if result is not None:
            return result
        event = self.generate_llm_event_based_on_history()
        return event

    def generate_llm_event_based_on_history(self):
        """
        基于历史记录生成事件
        """
        current_state = self.from_state
        # 如果app处于后台，则将app放置在前台
        event = self.move_the_app_to_foreground_if_needed(current_state)
        if event is not None:
            return event
        
        action, candidate_actions = self._get_action_with_LLM(current_state, self.__action_history, self.__activity_history)
        if action is not None:
            return action
        if self.__random_explore:
            self.logger.info("Trying random event...")
            action = random.choice(candidate_actions)
            return action
        # If couldn't find a exploration target, stop the app
        self.logger.info("Cannot find an exploration target. Trying to restart app...")
        self._event_trace += EVENT_FLAG_STOP_APP
        return KillAndRestartAppEvent(app=self.app)

    def _get_action_with_LLM(self, current_state, action_history, activity_history):
        """
        让大模型利用与局部探索相关的页面信息找到能够离开局部探索的路径，如果无法找到，则重启应用
        """
        task_prompt = (
                f"An application is being explored, and an action will be performed on a page to navigate to a new page.\n"
                f"The current exploration of the application has fallen into local exploration.\n"
                f"The description of the current local exploration:\n\"\"\"\n{self.description}\n\"\"\"\n\n"
                f"Your task is to generate the correct action to guide the current exploration away from local exploration.\n"
        )
        candidate_actions = current_state.get_candidate_actions()
        candidate_events_desc = []
        for candidate_action in candidate_actions:
            candidate_events_desc.append(f"- ({len(candidate_events_desc)}) {current_state.get_event_desc(candidate_action)}")
        info_prompt = (
            f"The description of the current page is:\n\"\"\"\n{activity_history[-1]}\n\"\"\"\n\n"
            f"The hierarchy semantic information on this page is:\n\"\"\"\n{current_state.get_semantic_info_tree()}\n\"\"\"\n\n"
            f"It has the following UI views and corresponding actions, with action id in parentheses:\n\"\"\"\n" + "\n".join(candidate_events_desc) + "\n\"\"\"\n\n"
        )
        output_prompt = (
            f"Determine which action shouble be performed to help the current exploration escape from the local exploration.\n"
            f"Give your answer and reason. The answer must be placed between labels <answer> </answer> and it must be the action id."
            f"If it is impossible to leave the current local exploration, it must be -1. The reason must be placed between labels <reason> </reason>.\n"
            f"This is two examples of the output content:\n"
            f"Example 1:\n\"\"\"\n"
            f"<answer>\n1\n</answer>\n"
            f"<reason>\nSince the current page is stuck in a popover, clicking the OK button is usually an effective way to leave the popover. So I choose action 1, which is to click the OK button.\n</reason>\n\"\"\"\n\n"
            f"Example 2:\n\"\"\"\n"
            f"<answer>\n4\n</answer>\n"
            f"<reason>\nThe exploration is currently stuck at the password‑setting function. The actions of clicking OK and clicking Reset password have failed to escape from the local exploration, so I must choose a different action instead of performing them again. Go back seems like a good choice, because it usually leaves a page, so I choose action 4, Go back.\n</reason>\n\"\"\"\n\n"
            f"Imitate the example and output the answer and reason. No other content is allowed.\n"
        )
        prompt = f"{task_prompt}{info_prompt}{output_prompt}"
        self.logger.info("Generating event based on LLM...")
        self.logger.info(f'[generate_llm_event] LLM prompt:\n{prompt}')
        
        response = self._query_llm(prompt).strip()
        
        # 提取answer标签内容
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        answer = answer_match.group(1).replace('\n', '') if answer_match else ""
        # 提取reason标签内容
        reason_pattern = r'<reason>(.*?)</reason>'
        reason_match = re.search(reason_pattern, response, re.DOTALL)
        reason = reason_match.group(1).replace('\n', '') if reason_match else ""
        
        if answer == "" or reason == "":
            self.logger.warning("[generate_llm_event] Invalid response from LLM. Unable to parse the answer and reason.")
            self.logger.warning(f"LLM response:\n{response}")

        # 正则匹配读取数字
        match = re.search(r"\d+", answer)
        if not match:
            self.logger.warning(f'[generate_llm_event] Invalid response from LLM. Unable to parse the answer and reason.')
            self.logger.warning(f"LLM response:\n{response}")
            # 如果解析失败，则随机生成事件
            self.logger.warning("Trying random event...")
            action = random.choice(candidate_actions)
            return action, candidate_actions
        idx = int(match.group(0))
        if idx == -1:
            self.logger.warning("LLM can not find an escape path. Trying to restart app...")
            self.logger.warning(f"The reason is:\n{reason}")
            return KillAndRestartAppEvent(app=self.app), candidate_actions
        # 检查idx是否超出范围
        if idx < -1 or idx >= len(candidate_actions):
            self.logger.warning(f"[generate_llm_event] Invalid response from LLM. The index is out of range.")
            self.logger.warning("Trying random event...")
            action = random.choice(candidate_actions)
            return action, candidate_actions
        # 获取选择的事件
        selected_action = candidate_actions[idx]
        self.logger.info(f"Generate event from LLM: {candidate_events_desc[idx]}")
        self.logger.info(f"The reason is:\n{reason}")
        
        # 额外处理输入文本
        if isinstance(selected_action, SetTextEvent):
            selected_action = self.text_generator(current_state, selected_action)
        return selected_action, candidate_actions

    def text_generator(self, current_state, action):
        view_desc = current_state.get_view_desc(action.view)
        task_prompt = f"Here is an editable view where you need to enter some text.\n"
        info_prompt = (
            f"The description of the page that the view belongs to is:\n\"\"\"\n{self.__activity_history[-1]}\n\"\"\"\n\n"
            f"The information about the view is:\n\"\"\"\n{view_desc}\n\"\"\"\n\n"
        )
        output_prompt = f"What text should be entered into it? Just return the text and nothing else.\n"
        prompt = f"{task_prompt}{info_prompt}{output_prompt}"
        self.logger.info("Generating input text...")
        self.logger.info(prompt)
        response = self._query_llm(prompt).strip()
        self.logger.info(f"Generate text for an editable view: {response}")
        action.text = response.replace('"', "")
        if len(action.text) > 30:  # heuristically disable long text input
            action.text = ""
        return action
    