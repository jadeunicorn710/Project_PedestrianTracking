"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        self.COV = 1. * np.eye(4)  # initialized covariance matrix
        self.D_t = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]])  # transition matrix
        self.M_t = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])  # measurement matrix
        self.Q = Q  # process noise matrix
        self.R = R  # measurement noise matrix

        self.K_t = np.zeros((4, 2))  # initialize kalman gain matrix
        self.Y_t = np.array([0, 0])  # initialize the measurement

        # raise NotImplementedError

    def predict(self):
        # predict state array
        self.state = self.D_t.dot(self.state)
        # predict covariance array
        self.COV = (self.D_t.dot(self.COV)).dot(np.transpose(self.D_t)) + self.Q


        # raise NotImplementedError

    def correct(self, meas_x, meas_y):
        # first initialize a 2x2 matrix to calculate the inverse matrix term
        K_inv = np.linalg.inv((self.M_t.dot(self.COV)).dot(np.transpose(self.M_t)) + self.R)
        # compute Kalman Gain
        self.K_t = (self.COV.dot(np.transpose(self.M_t))).dot(K_inv)

        # determine measurements Y_t
        self.Y_t[0] = meas_x
        self.Y_t[1] = meas_y

        # correct state X
        self.state = self.state + self.K_t.dot((self.Y_t - self.M_t.dot(self.state)))

        # correct covariance
        self.COV = (np.eye(4) - self.K_t.dot(self.M_t)).dot(self.COV)

        # raise NotImplementedError

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame

        # convert color template and frame to grayscale per assignment instruction
        self.template = 0.3 * self.template[:, :, 0] + 0.58 * self.template[:, :, 1] + 0.12 * self.template[:, :, 2]
        self.frame = 0.3 * self.frame[:, :, 0] + 0.58 * self.frame[:, :, 1] + 0.12 * self.frame[:, :, 2]

        self.particles = np.zeros((self.num_particles, 2))  # Initialize your particles array. Read the docstring.
        # take random sampling particles within frame and place in the particles array
        self.particles[:, 0] = np.random.choice(self.frame.shape[1], self.num_particles, True).astype(float)
        self.particles[:, 1] = np.random.choice(self.frame.shape[0], self.num_particles, True).astype(float)

        # Initialize with a uniform normalized distribution
        self.weights = (1 / self.num_particles) * np.ones(self.num_particles)  # Initialize your weights array. Read the docstring.

        # Initialize any other components you may need when designing your filter.
        # initialize the resampled particles array
        self.particles_resampled = np.zeros_like(self.particles)


        # raise NotImplementedError

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """

        # get m and n first
        m = template.shape[0]
        n = template.shape[1]

        # calculate MSE
        MSE = (1.0 / (m * n)) * np.sum((np.subtract(template, frame_cutout, dtype=np.float32)) ** 2)

        # calculate similarity value
        similarity = np.exp(-1 * MSE / (2.0 * self.sigma_exp ** 2))

        return similarity
        # return NotImplementedError

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """

        # first get a resampled index of the particles array
        index_resampled = np.random.choice(np.arange(self.num_particles), self.num_particles, True, self.weights)

        # loop through the particles array and place each particle into the resampled particles array
        for i in range(self.num_particles):
            self.particles_resampled[i, :] = self.particles[index_resampled[i], :]

        # limit particles within the frame
        self.particles_resampled[:, 0] = np.clip(self.particles_resampled[:, 0], 0, self.frame.shape[1] - 1)
        self.particles_resampled[:, 1] = np.clip(self.particles_resampled[:, 1], 0, self.frame.shape[0] - 1)

        return self.particles_resampled

        # return NotImplementedError

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        # convert the frame to grayscale per assignment instruction
        frame = 0.3 * frame[:, :, 0] + 0.58 * frame[:, :, 1] + 0.12 * frame[:, :, 2]

        # for dynamics, add normally distributed (gaussian) noise to each particle location
        sigma_t = np.random.normal(0, self.sigma_dyn, self.particles.shape)
        self.particles = self.particles + sigma_t

        # next need to get the updated weights to resample particles

        # limit the image patches of all particles within the frame
        # initialize an array to store all adjusted image patches
        frame_patches_adjusted = []

        # loop through all particles and add in eligible image patches
        for i in range(self.num_particles):
            # get the uppper left corner location of the image patch for the particle
            patch_left = int(self.particles[i, 0] - (self.template.shape[1]) / 2)
            # patch_right = patch_left + self.template.shape[1]
            patch_top = int(self.particles[i, 1] - (self.template.shape[0]) / 2)
            # patch_bottom = patch_top + self.template.shape[0]

            # limit patch within frame
            if patch_left < 0:
                patch_left = 0
            if patch_left + self.template.shape[1] > frame.shape[1] - 1:
                patch_left = frame.shape[1] - 1 - self.template.shape[1]
            if patch_top < 0:
                patch_top = 0
            if patch_top + self.template.shape[0] > frame.shape[0] - 1:
                patch_top = frame.shape[0] - 1 - self.template.shape[0]

            # add adjusted image patch to the patch array
            frame_patches_adjusted.append(frame[patch_top:patch_top + self.template.shape[0], patch_left:patch_left + self.template.shape[1]])

        # calculate weight for each particle
        for j in range(self.num_particles):
            self.weights[j] = self.get_error_metric(self.template, frame_patches_adjusted[j])

        # need to normalize the weight
        self.weights = self.weights / np.sum(self.weights)

        # now we can resample the particles
        self.particles = self.resample_particles()

        # raise NotImplementedError

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

            # plot each particle's location in the distribution in the frame
            cv2.circle(frame_in, (int(self.particles[i, 0]), int(self.particles[i, 1])), 1, (255, 255, 0), -1)

        # Complete the rest of the code as instructed.

        # define the start point and end point of the rectangle
        start_point = (int(x_weighted_mean - self.template.shape[1] / 2), int(y_weighted_mean - self.template.shape[0] / 2))
        end_point = (int(x_weighted_mean + self.template.shape[1] / 2), int(y_weighted_mean + self.template.shape[0] / 2))
        # draw the rectangle of the tracking window at the weighted mean
        cv2.rectangle(frame_in, start_point, end_point, (224, 255, 255), 2)

        # find the distance of every particle to the weighted mean and calculate the weighted sum

        # initialize the weighted sum
        distance_weighted_sum = 0

        # loop through particles and determine the distance to the weighted mean
        for i in range(self.num_particles):
            distance = np.sqrt((self.particles[i, 0] - x_weighted_mean) ** 2 + (self.particles[i, 1] - y_weighted_mean) ** 2)
            # add the weighted distance to weighted sum
            distance_weighted_sum = distance_weighted_sum + self.weights[i] * distance

        # plot a circle centered at the weighted mean with this radius
        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), int(distance_weighted_sum), (0, 0, 230), 1)

        # raise NotImplementedError


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """

        # convert the frame to grayscale per assignment instruction
        frame = 0.3 * frame[:, :, 0] + 0.58 * frame[:, :, 1] + 0.12 * frame[:, :, 2]

        # for dynamics, add normally distributed (gaussian) noise to each particle location
        sigma_t = np.random.normal(0, self.sigma_dyn, self.particles.shape)
        self.particles = self.particles + sigma_t

        # next need to get the updated weights to resample particles

        # limit the image patches of all particles within the frame
        # initialize an array to store all adjusted image patches
        frame_patches_adjusted = []

        # loop through all particles and add in eligible image patches
        for i in range(self.num_particles):
            # get the upper left corner location of the image patch for the particle
            patch_left = int(self.particles[i, 0] - (self.template.shape[1]) / 2)
            patch_top = int(self.particles[i, 1] - (self.template.shape[0]) / 2)

            # limit patch within frame
            if patch_left < 0:
                patch_left = 0
            if patch_left + self.template.shape[1] > frame.shape[1] - 1:
                patch_left = frame.shape[1] - 1 - self.template.shape[1]
            if patch_top < 0:
                patch_top = 0
            if patch_top + self.template.shape[0] > frame.shape[0] - 1:
                patch_top = frame.shape[0] - 1 - self.template.shape[0]

            # add adjusted image patch to the patch array
            frame_patches_adjusted.append(frame[patch_top:patch_top + self.template.shape[0], patch_left:patch_left + self.template.shape[1]])

        # calculate weight for each particle
        for j in range(self.num_particles):
            self.weights[j] = self.get_error_metric(self.template, frame_patches_adjusted[j])

        # need to normalize the weight
        self.weights = self.weights / np.sum(self.weights)

        # update the tracking window using IIR filter if alpha > 0
        if self.alpha > 0:
            # fist find the best tracking window for the current particle distribution
            window_center_x = self.particles[np.argmax(self.weights), 0]
            window_center_y = self.particles[np.argmax(self.weights), 1]

            # get the upper left corner location of the image patch for the particle
            window_left = int(window_center_x - (self.template.shape[1]) / 2)
            window_top = int(window_center_y - (self.template.shape[0]) / 2)

            # limit tracking window within frame
            if window_left < 0:
                window_left = 0
            if window_left + self.template.shape[1] > frame.shape[1] - 1:
                window_left = frame.shape[1] - 1 - self.template.shape[1]
            if window_top < 0:
                window_top = 0
            if window_top + self.template.shape[0] > frame.shape[0] - 1:
                window_top = frame.shape[0] - 1 - self.template.shape[0]

            # get the adjusted image patch
            frame_patch_adjusted = frame[window_top:window_top + self.template.shape[0], window_left:window_left + self.template.shape[1]]

            # update the current window model to be a weighted sum of the last model and the current best estimate
            self.template = self.alpha * frame_patch_adjusted + (1 - self.alpha) * self.template

        # now we can resample the particles
        self.particles = ParticleFilter.resample_particles(self)


        # raise NotImplementedError


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        # initialize the frame number
        self.frame_number = 1

        # make a duplicate of the grayscale template
        self.template_original = 0.3 * template[:, :, 0] + 0.58 * template[:, :, 1] + 0.12 * template[:, :, 2]

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        # convert the frame to grayscale per assignment instruction
        frame = 0.3 * frame[:, :, 0] + 0.58 * frame[:, :, 1] + 0.12 * frame[:, :, 2]

        # next need to get the updated weights

        # limit the image patches of all particles within the frame
        # initialize an array to store all adjusted image patches
        frame_patches_adjusted = []

        # loop through all particles and add in eligible image patches
        for i in range(self.num_particles):
            # get the uppper left corner location of the image patch for the particle
            patch_left = int(self.particles[i, 0] - (self.template.shape[1]) / 2)
            # patch_right = patch_left + self.template.shape[1]
            patch_top = int(self.particles[i, 1] - (self.template.shape[0]) / 2)
            # patch_bottom = patch_top + self.template.shape[0]

            # limit patch within frame
            if patch_left < 0:
                patch_left = 0
            if patch_left + self.template.shape[1] > frame.shape[1] - 1:
                patch_left = frame.shape[1] - 1 - self.template.shape[1]
            if patch_top < 0:
                patch_top = 0
            if patch_top + self.template.shape[0] > frame.shape[0] - 1:
                patch_top = frame.shape[0] - 1 - self.template.shape[0]

            # add adjusted image patch to the patch array
            frame_patches_adjusted.append(frame[patch_top:patch_top + self.template.shape[0], patch_left:patch_left + self.template.shape[1]])

        # calculate weight for each particle
        for j in range(self.num_particles):
            self.weights[j] = self.get_error_metric(self.template, frame_patches_adjusted[j])

        # need to normalize the weight
        self.weights = self.weights / np.sum(self.weights)

        # get the maximum weight to check for occlusion existence, stop resampling if occlusion occurs,
        # instead, manually update particle distribution
        max_weight = self.weights[np.argmax(self.weights)]

        # first occlusion occurs between frame number 123 - 153
        if (self.frame_number >= 123 and self.frame_number <= 153):
            if max_weight >= 0.01:
                self.particles[:, 0] = self.particles[:, 0] - 0.04

        # second occlusion occurs between frame number 185 - 204
        elif (self.frame_number >= 185 and self.frame_number <= 204):
            if max_weight >= 0.01:
                self.particles[:, 0] = self.particles[:, 0] - 0.2

        # use random noise and resample if no occlusion exists
        else:
            # for dynamics, add normally distributed (gaussian) noise to each particle location
            sigma_t = np.random.normal(0, self.sigma_dyn, self.particles.shape)
            self.particles = self.particles + sigma_t
            # now we can resample the particles
            self.particles = self.resample_particles()

        # need to resize the template for tracking
        self.template = cv2.resize(self.template_original, (0, 0), fx=(0.995 ** self.frame_number), fy=(0.995 ** self.frame_number))

        # update the frame number
        self.frame_number += 1

        # raise NotImplementedError