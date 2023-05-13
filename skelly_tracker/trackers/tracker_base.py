from abc import ABC, abstractmethod

class TrackerBase(ABC):
    
    @abstractmethod
    def process_image(self, image):
        """
        Process the input image and apply the tracking algorithm.
        
        :param image: An input image.
        :return: None
        """
        pass

    @abstractmethod
    def get_output_data(self):
        """
        Retrieve the tracking data after processing an image.
        
        :return: A dictionary containing the tracking data.
        """
        pass

    @abstractmethod
    def get_annotated_image(self):
        """
        Retrieve the annotated image after processing an image.
        
        :return: An image with annotations.
        """
        pass
