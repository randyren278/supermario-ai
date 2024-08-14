from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import cv2
import numpy as np

class GameEnvironment:
    def __init__(self, driver_path):
        self.driver_path = driver_path
        self.driver = webdriver.Chrome(executable_path=self.driver_path)
        self.driver.get('https://jcw87.github.io/c2-smb1/')
        time.sleep(5)  # Allow the game to load

    def press_key(self, key):
        game_canvas = self.driver.find_element_by_id('canvas')
        game_canvas.send_keys(key)

    def get_screenshot(self):
        screenshot = self.driver.get_screenshot_as_png()
        screenshot = np.frombuffer(screenshot, np.uint8)
        img = cv2.imdecode(screenshot, cv2.IMREAD_COLOR)
        return img

    def close(self):
        self.driver.quit()

# Example of usage
if __name__ == "__main__":
    env = GameEnvironment(driver_path='/path/to/chromedriver')
    env.press_key(Keys.ARROW_RIGHT)  # Example of pressing the right arrow key
    time.sleep(2)
    screenshot = env.get_screenshot()
    cv2.imwrite('screenshot.png', screenshot)
    env.close()
