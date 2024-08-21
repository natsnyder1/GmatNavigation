import sys, os, math
import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
gmat = None

MonthDic = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
MonthDic2 = {v:k for k,v in MonthDic.items()}

# Converts a datetime object to a string
def datetime_2_time_string(t):
    global MonthDic2
    t_format = "{} {} {} ".format(t.day, MonthDic2[t.month], t.year)
    if t.hour < 10:
        t_format += "0"
    t_format += str(t.hour) + ":"
    if t.minute < 10:
        t_format += "0"
    t_format += str(t.minute) + ":"
    if t.second < 10:
        t_format += "0"
    t_format += str(t.second) + "."
    millisec = str(np.round(t.microsecond / 1e6, 3)).split(".")[1]
    if millisec == "0":
        millisec = "000"
    t_format += millisec
    return t_format 

# Converts a string to a datetime object
def time_string_2_datetime(t):
    return datetime.strptime(t, "%d %b %Y %H:%M:%S.%f")

# Time conversion function using GMAT
# time_in: can either be a string (Gregorian), datetime (Gregorian), or float (MJD)
# type_in: can be one of "A1", "TAI", "UTC", "TDB", "TT"
# type_out: can be one of "A1", "TAI", "UTC", "TDB", "TT"
# returns a dictionary populated as 
#  {
#   "in_greg" : time_in_greg -> Your Input Time (of type_in) In Gregorian Format
#   "in_mjd" : time_in_greg -> Your Input Time (of type_in) In MJD Format
#   "out_greg": time_out_greg -> Your Output Time (of type_out) In Gregorian Format
#   "out_mjd": time_out_mjd -> Your Output Time (of type_out) In MJD Format
#  }
def time_convert(time_in, type_in, type_out):
    if type(time_in) == datetime:
        millisec = str(np.round(time_in.microsecond / 1e6, 3)).split(".")[1]
        _time_in = time_in.strftime("%d %b %Y %H:%M:%S") + "." + millisec
        is_in_gregorian = True
    elif type(time_in) == str:
        _time_in = time_in
        is_in_gregorian = True
    elif type(time_in) == float:
        _time_in = time_in
        is_in_gregorian = False
    else:
        print("Time In Type: ", type(time_in), " Not Supported! Input was", time_in)
        exit(1)
    timecvt = gmat.TimeSystemConverter.Instance()
    if is_in_gregorian:
        time_in_greg = _time_in
        time_in_mjd = timecvt.ConvertGregorianToMjd(_time_in)
    else:
        time_in_mjd = _time_in
        time_in_greg = timecvt.ConvertMjdToGregorian(_time_in)
    time_types = {"A1": timecvt.A1, "TAI": timecvt.TAI, "UTC" : timecvt.UTC, "TDB": timecvt.TDB, "TT": timecvt.TT}
    assert type_in in time_types.keys()
    assert type_out in time_types.keys()
    time_code_in = time_types[type_in]
    time_code_out = time_types[type_out]
    time_out_mjd = timecvt.Convert(time_in_mjd, time_code_in, time_code_out)
    time_out_greg = timecvt.ConvertMjdToGregorian(time_out_mjd)
    time_dic = {"in_greg" : time_in_greg, 
                "in_mjd" : time_in_mjd, 
                "out_greg": time_out_greg, 
                "out_mjd": time_out_mjd}
    return time_dic

# Draw a random exponential variable of the pdf \lambda * exp(-\lambda * x)
def random_exponential(lam):
    EPS = 1e-16
    ALMOST_ONE = 1.0 - EPS
    # Draw a random uniform variable on the open interval (0,1)
    U = np.random.uniform(EPS, ALMOST_ONE)
    return  -np.log( U ) / lam

# Draw a random alpha stable variable 
# This function assumes that the beta (skewness parameter) for the random alpha stable method is zero 
# the parameters are: 
# 1.) alpha \in (0,2] -- this is the stability param (2=Gaussian, 1 = Cauchy, 0.5 = Levy)
# 2.) c \in (0, inf] -- this is the scale param (standard deviation for Gaussian)
# 3.) mu \in [-inf,inf] -- this is the location parameter
# Note: For any value of alpha less than or equal to 2, the variance is undefined 
# This implements the Chambers, Mallows, and Stuck (CMS) method from their seminal paper in 1976
def random_symmetric_alpha_stable(alpha, c, mu):
    EPS = 1e-16
    ALMOST_ONE = 1.0 - EPS
    #Generate a random variable on interval (-pi/2, pi/2)
    U = np.random.uniform(-np.pi/2.0, np.pi/2.0) * ALMOST_ONE
    #Generate a random exponential variable with mean of 1.0
    W = random_exponential(1.0)
    xi = np.pi / 2.0 if alpha == 1.0 else 0.0
    X = 0.0 # ~ S_\alpha(\beta,1,0)
    if(alpha == 1.0):
        X = np.tan(U)
    else:
        X = np.sin(alpha*(U+xi)) / (np.cos(U)**(1.0/alpha)) * ((np.cos(U - alpha*(U + xi))) / W)**( (1.0 - alpha) / alpha )
    # Now scale and locate the random variable
    Y = c*X + mu
    return Y

# Process Noise Model for position and velocity
def leo6_process_noise_model(dt, qs):
    if type(qs) == float:
        qs = np.repeat(qs, 3)
    elif type(qs) == list:
        qs = np.array(qs)
    if qs.size == 1:
        qs = np.repeat(qs, 3)
    if qs.size != 3:
        print("Error leo6_process_noise_model: The power spectral density for acceleration noise: qs={} is not a 3-vector!\nPlease either provide a 3-vector, or provide a single value to be used for all three accelerations! Exiting!".format(qs))
        exit(1)
    W = np.zeros((6,6))
    W[0:3,0:3] = np.diag(qs) * dt**3 / 3 
    W[0:3,3:6] = np.diag(qs) * dt**2 / 2 
    W[3:6,0:3] = np.diag(qs) * dt**2 / 2 
    W[3:6,3:6] = np.diag(qs) * dt 
    return W

def check_then_load_gmatpy(gmat_root_dir, eop_filepath, spaceweather_filepath):
    # --------------- AUTOMATED CHECKS ON USER CONFIGURED PATHS ----------------- #
    # 1.) Test directory location of the SpaceWeather and EOP File for existence
    if not os.path.isfile(eop_filepath):
        print("ERROR: eop_filepath: {}\nSEEN NOT TO EXIST! Please check path! Exiting!".format(eop_filepath))
        exit(1)
    if not os.path.isfile(spaceweather_filepath):
        print("ERROR: spaceweather_filepath: {}\nSEEN NOT TO EXIST! Please check path! Exiting!".format(spaceweather_filepath))
        exit(1)
    # 2.) Check GMAT Subdirectories for existence, leaving nothing to chance
    gmat_application_dir = gmat_root_dir + '/application' 
    if os.path.exists(gmat_application_dir):
        # GMAT's bin, api, and python interface (gmatpy) directories, and the location of api_startup_file.txt, respectively
        gmat_bin_dir = gmat_application_dir + '/bin'
        gmat_api_dir = gmat_application_dir + '/api'
        gmat_py_dir = gmat_bin_dir + '/gmatpy'
        gmat_startup_file = gmat_bin_dir + '/api_startup_file.txt'
        if os.path.exists(gmat_bin_dir):
            if gmat_bin_dir not in sys.path:
                sys.path.append(gmat_bin_dir)
        else:
            print("ERROR: gmat_bin_dir: {}\nSEEN NOT TO EXIST! Please check path! Exiting!".format(gmat_bin_dir))
            exit(1)
        if os.path.exists(gmat_api_dir):
            if gmat_api_dir not in sys.path:
                sys.path.append(gmat_api_dir)
        else:
            print("ERROR: gmat_api_dir: {}\nSEEN NOT TO EXIST! Please check path! Exiting!".format(gmat_api_dir))
            exit(1)
        if os.path.exists(gmat_py_dir):
            if gmat_py_dir not in sys.path:
                sys.path.append(gmat_py_dir)
        else:
            print("ERROR: gmat_py_dir: {}\nSEEN NOT TO EXIST! Please check path! Exiting!".format(gmat_py_dir))
            exit(1)
        if os.path.exists(gmat_startup_file):
            import gmat_py as _gmat # The main act
            _gmat.Setup(gmat_startup_file) # The main act
            global gmat
            gmat = _gmat
        else:
            print("ERROR: gmat_startup_file: {}\nSEEN NOT TO EXIST! Please check path! Exiting!".format(gmat_startup_file))
            exit(1)
    else:
        print("ERROR: gmat_application_dir: {}\nSEEN NOT TO EXIST! Please check path! Exiting!".format(gmat_application_dir))
        exit(1)
    print("Successfully loaded GMAT Python module gmat_py!")

class EarthOrbitingSatellite():
    def __init__(self, eop_filepath, spaceweather_filepath, gmat_print=False):
        
        self.eop_filepath = eop_filepath
        self.spaceweather_filepath = spaceweather_filepath
        self.gmat_print = gmat_print
        if not os.path.isfile(self.eop_filepath):
            print("ERROR CreateSatellite: EOP Filepath {}\nDoes Not Exist! Please Check Path! Exiting!".format(eop_filepath))
            exit(1)
        if not os.path.isfile(self.spaceweather_filepath):
            print("ERROR CreateSatellite: SpaceWeather Filepath {}\nDoes Not Exist! Please Check Path! Exiting!".format(spaceweather_filepath))
            exit(1)

        # Solve for list 
        self.solve_for_states = [] 
        self.solve_for_taus = []
        self.solve_for_fields = [] 
        self.solve_for_dists = [] 
        self.solve_for_scales = [] 
        self.solve_for_nominals = []
        self.solve_for_alphas = []
        self.solve_for_fields_acceptable = ["Cd", "Cr"] # Can be enlarged
        self.solve_for_dists_acceptable = ["gauss", "sas"]
        self.is_model_constructed = False

    def assert_model_uncontructed(self, func_str):
        if self.is_model_constructed:
            print("Error EarthOrbitingSatellite.{}(...)\n  Model is already constructed!\n  Please use the .clear_model() method to tear previous model down before building a new model! Exiting!".format(func_str))
            exit(1)
    def assert_model_contructed(self, func_str):
        if not self.is_model_constructed:
            print("Error EarthOrbitingSatellite.{}(...)\n  Model is not constructed!\n  Please use the .construct_model() method to build a model down before continuing! Exiting!".format(func_str))
            exit(1)
    def assert_x0_size(self, func_str):
        if self.x0.size != 6:
            print("Error EarthOrbitingSatellite.{}(...)\n  x0 is not a vector of length six!\n  Please look again at x0, which should be x0 = np.array([pox_x, pox_y, pox_z, vel_x, vel_y, vel_z])!".format(func_str))
            exit(1)
    def assert_x_size(self, func_str, x, size_x_req):
        if x.size != size_x_req:
            print("Error EarthOrbitingSatellite.{}(...)\n  len(x)={} but should be a vector of length {}!\n  Please look again at x, which should have len(x) = len(6) + len(your_solve_fors)!".format(func_str, x.size(), size_x_req))
            exit(1)

    def create_model(self, t0, x0, Cd0, Cr0, A, m, fuel_mass, solar_flux, dt,
            earth_model_degree = 70,
            earth_model_order = 70,
            with_jacchia = True,
            with_SRP = True, 
            integrator_initial_step_size = None,
            integrator_accuracy = 1e-13,
            integrator_min_step = 0,
            integrator_max_step = None,
            integrator_max_step_attempts = 50):
        self.assert_model_uncontructed("create_model")

        if type(t0) == str:
            self.t0 = t0
            self.t0_datetime = time_string_2_datetime(t0)
        elif type(t0) == datetime:
            self.t0 = datetime_2_time_string(t0)
            self.t0_datetime = t0
        else:
            print("Unrecognized format for starting time t0! Must be string or datetime object! Exiting!")
            exit(1)
        self.x0 = x0.copy()
        self.dt = dt 
        self.assert_x0_size("create_model")

        # Solar System Properties -- Newly Added
        mod = gmat.Moderator.Instance()
        self.ss = mod.GetDefaultSolarSystem()
        self.ss.SetField("EphemerisSource", "DE421")
        self.earth = self.ss.GetBody('Earth')
        self.earth.SetField('EopFileName', self.eop_filepath)

        # Create Fermi Model
        self.sat = gmat.Construct("Spacecraft", "Fermi")
        self.sat.SetField("DateFormat", "UTCGregorian")
        self.sat.SetField("CoordinateSystem", "EarthMJ2000Eq")
        self.sat.SetField("DisplayStateType","Cartesian") 
    
        self.sat.SetField("Epoch", self.t0) 
        self.sat.SetField("DryMass", m)
        self.sat.SetField("Cd", Cd0)
        self.sat.SetField("Cr", Cr0)
        #self.sat.SetField("CrSigma", 0.1)
        self.sat.SetField("DragArea", A)
        self.sat.SetField("SRPArea", A)
        self.sat.SetField("Id", '2525')
        self.sat.SetField("X", self.x0[0])
        self.sat.SetField("Y", self.x0[1])
        self.sat.SetField("Z", self.x0[2])
        self.sat.SetField("VX", self.x0[3])
        self.sat.SetField("VY", self.x0[4])
        self.sat.SetField("VZ", self.x0[5])

        self.fueltank = gmat.Construct("ChemicalTank", "FuelTank")
        self.fueltank.SetField("FuelMass", fuel_mass) 
        if self.gmat_print:
            self.fueltank.Help()
        self.sat.SetField("Tanks", "FuelTank") 
        if self.gmat_print:
            self.sat.Help()
            print(self.sat.GetGeneratingString(0))

        # Create Force Model 
        self.fm = gmat.Construct("ForceModel", "TheForces")
        self.fm.SetField("ErrorControl", "None")
        # A 70x70 EGM96 Gravity Model
        self.earthgrav = gmat.Construct("GravityField")
        self.earthgrav.SetField("BodyName","Earth")
        self.earthgrav.SetField("Degree", earth_model_degree)
        self.earthgrav.SetField("Order", earth_model_order)
        self.earthgrav.SetField("PotentialFile","EGM96.cof")
        self.earthgrav.SetField("TideModel", "SolidAndPole")
        # The Point Masses
        self.moongrav = gmat.Construct("PointMassForce")
        self.moongrav.SetField("BodyName","Luna")
        self.sungrav = gmat.Construct("PointMassForce")
        self.sungrav.SetField("BodyName","Sun")
        # Solar Radiation Pressure
        if with_SRP:
            self.srp = gmat.Construct("SolarRadiationPressure")
            self.srp.SetField("SRPModel", "Spherical")
            self.srp.SetField("Flux", solar_flux)
        # Drag Model
        if with_jacchia:
            self.jrdrag = gmat.Construct("DragForce")
            self.jrdrag.SetField("AtmosphereModel","JacchiaRoberts")
            self.jrdrag.SetField("HistoricWeatherSource", 'CSSISpaceWeatherFile')
            self.jrdrag.SetField("CSSISpaceWeatherFile", self.spaceweather_filepath)
            # Build and set the atmosphere for the model
            self.atmos = gmat.Construct("JacchiaRoberts")
            self.jrdrag.SetReference(self.atmos)

        self.fm.AddForce(self.earthgrav)
        self.fm.AddForce(self.moongrav)
        self.fm.AddForce(self.sungrav)
        if with_jacchia:
            self.fm.AddForce(self.jrdrag)
        if with_SRP:
            self.fm.AddForce(self.srp)
        
        if self.gmat_print:
            self.fm.Help()
            print(self.fm.GetGeneratingString(0))

        # Build Integrator
        self.gator = gmat.Construct("RungeKutta89", "Gator")
        # Build the propagation container that connect the integrator, force model, and spacecraft together
        self.pdprop = gmat.Construct("Propagator","PDProp")  
        # Create and assign a numerical integrator for use in the propagation
        self.pdprop.SetReference(self.gator)
        # Set some of the fields for the integration
        if integrator_initial_step_size is None:
            integrator_initial_step_size = self.dt 
        if integrator_max_step is None:
            integrator_max_step = self.dt
        self.pdprop.SetField("InitialStepSize", integrator_initial_step_size)
        self.pdprop.SetField("Accuracy", integrator_accuracy)
        self.pdprop.SetField("MinStep", integrator_min_step)
        self.pdprop.SetField("MaxStep", integrator_max_step)
        self.pdprop.SetField("MaxStepAttempts", integrator_max_step_attempts)

        # Assign the force model to the propagator
        self.pdprop.SetReference(self.fm)
        # It also needs to know the object that is propagated
        self.pdprop.AddPropObject(self.sat)
        # Setup the state vector used for the force, connecting the spacecraft
        self.psm = gmat.PropagationStateManager()
        self.psm.SetObject(self.sat)
        self.psm.SetProperty("AMatrix")
        self.psm.BuildState()
        # Finish the object connection
        self.fm.SetPropStateManager(self.psm)
        self.fm.SetState(self.psm.GetState())

        # Before Initialization, make two coordinate converters to help convert MJ2000Eq Frame to ECF Frame
        self.ecf = gmat.Construct("CoordinateSystem","ECF","Earth","BodyFixed")
        self.eci = gmat.Construct("CoordinateSystem","ECI","Earth","MJ2000Eq")
        self.csConverter = gmat.CoordinateConverter()

        # Perform top level initialization
        gmat.Initialize()

        # Finish force model setup:
        #  Map the spacecraft state into the model
        self.fm.BuildModelFromMap()
        #  Load the physical parameters needed for the forces
        self.fm.UpdateInitialData()
        # Perform the integation subsysem initialization
        self.pdprop.PrepareInternals()
        # Refresh the integrator reference
        self.gator = self.pdprop.GetPropagator()
        self.is_model_constructed = True

    def clear_model(self):
        gmat.Clear()
        self.is_model_constructed = False

    # defaults to current time if not specified
    def reset_state(self, x, iter = None):
        if iter is None: # iter is the "iteration"
            ellapsed_time = self.get_ellapsed_time()
        else:
            assert iter >= 0.0
            ellapsed_time = iter * self.dt
        self.assert_x_size("reset_state", x, 6 + len(self.solve_for_states))
        for j in range(len(self.solve_for_states)):
            self.solve_for_states[j] = x[6+j]
            val = self.solve_for_nominals[j] * ( 1 + self.solve_for_states[j] )
            val = np.clip(val, -.999, np.inf) # Make sure value is valid for GMAT
            self.sat.SetField(self.solve_for_fields[j], val)
        self.sat.SetState(*x[0:6])
        self.fm.BuildModelFromMap()
        self.fm.UpdateInitialData()
        self.pdprop.PrepareInternals()
        self.gator = self.pdprop.GetPropagator() # refresh integrator
        self.gator.SetTime(ellapsed_time)
    
    # defaults to current time if not specified
    def reset_state_with_ellapsed_time(self, x, ellapsed_time = None):
        self.assert_x_size("reset_state", x, 6 + len(self.solve_for_states))
        if ellapsed_time is None:
            ellapsed_time = self.get_ellapsed_time()
        else:
            assert ellapsed_time >= 0.0
        for j in range(len(self.solve_for_states)):
            self.solve_for_states[j] = x[6+j]
            val = self.solve_for_nominals[j] * ( 1 + self.solve_for_states[j] )
            val = np.clip(val, -.999, np.inf) # Make sure value is valid for GMAT
            self.sat.SetField(self.solve_for_fields[j], val)
        self.sat.SetState(*x[0:6])
        self.fm.BuildModelFromMap()
        self.fm.UpdateInitialData()
        self.pdprop.PrepareInternals()
        self.gator = self.pdprop.GetPropagator() # refresh integrator
        self.gator.SetTime(ellapsed_time)

    def solve_for_state_jacobians(self, Jac, dv_dt):
        # Nominal dv_dt w/out parameter changes is inputted
        pstate = self.gator.GetState()
        eps = 0.0005
        for j in range(len(self.solve_for_states)):
            val = float( self.sat.GetField(self.solve_for_fields[j]) )
            # dv_dt with small parameter change
            self.sat.SetField(self.solve_for_fields[j], val+eps)
            self.fm.GetDerivatives(pstate, dt=self.dt, order=1)
            dv_eps_dt = np.array(self.fm.GetDerivativeArray()[3:6])
            Jac_sfs = (dv_eps_dt - dv_dt) / (eps/self.solve_for_nominals[j]) # Simple Model -- Can make derivative more robust
            Jac[3:6, 6+j] = Jac_sfs
            Jac[6+j, 6+j] = -1.0 / self.solve_for_taus[j]
            # reset nominal field
            self.sat.SetField(self.solve_for_fields[j], val)
        return Jac

    def get_jacobian_matrix(self):
        pstate = self.gator.GetState() #sat.GetState().GetState()
        self.fm.GetDerivatives(pstate, dt=self.dt, order=1)
        fdot = self.fm.GetDerivativeArray()
        dx_dt = np.array(fdot[0:6])
        num_sf = len(self.solve_for_states)
        num_x = 6 + num_sf
        Jac = np.zeros((num_x, num_x))
        # Add Position and Velocity State Jacobians 
        Jac[0:6,0:6] = np.array(fdot[6:42]).reshape((6,6))
        # Add solve for state Jacobians
        if num_sf > 0:
            Jac = self.solve_for_state_jacobians(Jac, dx_dt[3:])
        return Jac
    
    def get_ellapsed_time(self):
        return self.gator.GetTime()
    
    def _compute_STM(self, Jac, dt, power_order, convert_Jac_to_meters = False):
        n = Jac.shape[0]
        if convert_Jac_to_meters:
            Jac[3:6,6] *= 1000 # convert Jac to meter-based Jacobian
        Phi = np.eye(n) + Jac * dt
        for i in range(2, power_order+1):
            Phi += np.linalg.matrix_power(Jac, i) * dt**i / math.factorial(i)
        return Phi

    def get_transition_matrix(self, power_order, num_sub_steps = None, with_avg_jac = True, use_units_km = True):
        if num_sub_steps is None:
            num_sub_steps = 1.0 
        else:
            assert(num_sub_steps > 0)
        num_sub_steps = int(num_sub_steps)
        dt = self.dt
        x0 = self.get_state()
        ellapsed_time = self.get_ellapsed_time()
        dt_sub = dt / num_sub_steps
        self.dt = dt_sub
        n = 6 + len(self.solve_for_nominals)
        if with_avg_jac:
            STM_AVG_JAC = np.eye(n)
            for i in range(num_sub_steps):
                # Get Jacobians and STMs over time step DT_SUB
                Jac_i = self.get_jacobian_matrix()
                self.step()
                Jac_ip1 = self.get_jacobian_matrix()
                Jac_avg = (Jac_i+Jac_ip1)/2
                if not use_units_km:
                    Jac_avg[3:6,6:] *= 1000
                STM_AVG_JAC = self._compute_STM(Jac_avg, dt_sub, power_order) @ STM_AVG_JAC
            self.dt = dt
            self.reset_state_with_ellapsed_time(x0, ellapsed_time)
            return STM_AVG_JAC
        else:
            STM = np.eye(n)
            for i in range(num_sub_steps):
                # Get Jacobians and STMs over time step DT_SUB
                Jac_i = self.get_jacobian_matrix()
                if num_sub_steps > 1:
                    self.step()
                if not use_units_km:
                    Jac_i[3:6,6:] *= 1000
                STM = self._compute_STM(Jac_i, dt_sub, power_order) @ STM
            self.dt = dt
            if num_sub_steps > 1:
                self.reset_state_with_ellapsed_time(x0, ellapsed_time)
            return STM

    def step(self, noisy_prop_solve_for = False):
        self.gator.Step(self.dt)
        num_sf = len(self.solve_for_states)
        xk = np.zeros(6 + num_sf)
        xk[0:6] = np.array(self.gator.GetState())
        if (num_sf > 0):
            if noisy_prop_solve_for:
                xk[6:], wk = self.propagate_solve_fors(noisy_prop_solve_for)
                return xk, wk 
            else:
                xk[6:] = self.propagate_solve_fors(noisy_prop_solve_for)
                return xk
        else:
            return xk

    def get_state(self):
        return np.array(self.gator.GetState() + self.solve_for_states)

    def get_state6_derivatives(self):
        pstate = self.gator.GetState()
        self.fm.GetDerivatives(pstate, dt=self.dt, order=1) 
        fdot = self.fm.GetDerivativeArray()
        dx_dt = np.array(fdot[0:6])
        return dx_dt
    
    # Solve For State Methods
    def set_solve_for(self, field = "Cd", dist="gauss", scale = -1, tau = -1, alpha = None):
        self.assert_model_contructed("set_solve_for")
        assert field in self.solve_for_fields_acceptable 
        assert dist in self.solve_for_dists_acceptable 
        assert scale > 0 
        assert tau > 0
        if dist == "sas":
            assert alpha is not None 
            assert alpha >= 1
        if field not in self.solve_for_fields:
            self.solve_for_states.append(0.0)
            self.solve_for_taus.append(tau)
            self.solve_for_fields.append(field)
            self.solve_for_dists.append(dist)
            self.solve_for_scales.append(scale)
            self.solve_for_alphas.append(alpha)
            self.solve_for_nominals.append( float( self.sat.GetField(field) ) )
        else:
            print("Solve For State: {} is already in the list of solve fors! Not Reappending! Recheck code -- you can possibly cause errors in your simulation (the initial value of the solve for may be incorrect, just create a new object...). Exiting for now!".format(field))
            exit(1)

    def get_solve_for_noise_sample(self, j):
        if self.solve_for_dists[j] == "gauss":
            return np.random.randn() * self.solve_for_scales[j]
        elif self.solve_for_dists[j] == "sas":
            return random_symmetric_alpha_stable(self.solve_for_alphas[j], self.solve_for_scales[j], 0)
        else:
            print( "Solve for distribution {} has not been implemented in get_solve_for_noise_sample() function! Please add it!".format(self.solve_for_dists[j]) )
            exit(1)
    
    def propagate_solve_fors(self, with_add_noise = False):
        new_sf_states = []
        noises = []
        for j in range(len(self.solve_for_states)):
            tau = self.solve_for_taus[j]
            self.solve_for_states[j] = np.exp(-self.dt / tau) * self.solve_for_states[j]
            if with_add_noise:
                noise = self.get_solve_for_noise_sample(j)
                self.solve_for_states[j] += noise
                self.solve_for_states[j] = np.clip(self.solve_for_states[j], -0.99, np.inf) # cannot be <= -1
                noises.append(noise)
            new_sf_states.append( self.solve_for_states[j] )
            new_nom_val = self.solve_for_nominals[j] * (1 + self.solve_for_states[j])
            self.sat.SetField(self.solve_for_fields[j], new_nom_val)
        if with_add_noise:
            return new_sf_states, noises
        else:
            return new_sf_states

    # Simulation Methods
    def simulate(self, num_props, gps_std_dev, W=None):
        self.assert_model_contructed("simulate")
        self.gps_std_dev = gps_std_dev 
        with_solve_fors = len(self.solve_for_states) > 0
        with_sim_state_reset = (W is not None) or with_solve_fors
        # Measurement before propagation
        len_x = 6 + len(self.solve_for_states)
        x0 = np.array( list(self.x0) + self.solve_for_states )
        v0 = np.random.randn(3) * self.gps_std_dev
        z0 = self.transform_Earth_MJ2000Eq_2_BodyFixed()[0:3] + v0
        states = [x0]
        msmt_noises = [v0]
        msmts = [ z0 ]
        proc_noises = []
        # Begin loop for propagation
        for i in range(num_props):
            wk = np.zeros(len_x)
            xk = np.zeros(len_x)
            # Now step the integrator and get new state
            if with_solve_fors:
                xk, wk[6:] = self.step(noisy_prop_solve_for = True)
            else:
                xk = self.step(noisy_prop_solve_for = False)
            # Add process noise to pos/vel, if given
            if W is not None:
                noise = np.random.multivariate_normal(np.zeros(6), W)
                xk[0:6] += noise
                wk[0:6] = noise
            # If solve fors are declared, or process noise is added, need to update state of simulator
            if with_sim_state_reset:
                self.reset_state(xk, i+1)
            # Append State and Process Noises
            proc_noises.append(wk)
            states.append(xk)
            # Form measurement
            vk = np.random.randn(3) * self.gps_std_dev
            zk = self.transform_Earth_MJ2000Eq_2_BodyFixed()[0:3] + vk
            msmts.append(zk)
            msmt_noises.append(vk)
        
        # Reset Solve Fors
        for j in range(len(self.solve_for_states)):
            self.solve_for_states[j] = 0.0
            self.sat.SetField(self.solve_for_fields[j], self.solve_for_nominals[j])

        # Reset Simulation to x0, and return state info
        self.reset_state(x0, 0)
        return np.array(states), np.array(msmts), np.array(proc_noises), np.array(msmt_noises)

    def transform_Earth_MJ2000Eq_2_BodyFixed(self, state_mj2000 = None, time_a1mjd = None):
        # If a time is not explicitly given, use current epoch
        if time_a1mjd is None:
            t_datetime = self.t0_datetime + timedelta(self.get_ellapsed_time())
            time_a1mjd = time_convert(t_datetime, "UTC", "A1")["out_mjd"]
        if state_mj2000 is None:
            state_mj2000 = self.get_state()
        elif state_mj2000.size == 3:
            state_mj2000 = np.concatenate((state_mj2000,np.zeros(3)))
        state_in = gmat.Rvector6(*list(state_mj2000[0:6])) # State in MJ2000Eq
        state_out = gmat.Rvector6() # State in Earth Body Fixed Coordinates
        self.csConverter.Convert(time_a1mjd, state_in, self.eci, state_out, self.ecf)
        so_array = np.array([ state_out[0], state_out[1], state_out[2], state_out[3], state_out[4], state_out[5] ])
        return so_array # ECF (Earth Coordinates Fixed)
    
    def transform_Earth_BodyFixed_2_MJ2000Eq(self, state_earth_bf, time_a1mjd = None):
        # If a time is not explicitly given, use current epoch
        if time_a1mjd is None:
            t_datetime = self.t0_datetime + timedelta(self.get_ellapsed_time())
            time_a1mjd = time_convert(t_datetime, "UTC", "A1")["out_mjd"]
        if state_earth_bf.size == 3:
            state_earth_bf = np.concatenate((state_earth_bf,np.zeros(3)))
        state_in = gmat.Rvector6(*list(state_earth_bf[0:6])) # State in Earth Body Fixed Coordinates
        state_out = gmat.Rvector6() # State in MJ2000Eq
        self.csConverter.Convert(time_a1mjd, state_in, self.ecf, state_out, self.eci)
        so_array = np.array([ state_out[0], state_out[1], state_out[2], state_out[3], state_out[4], state_out[5] ])
        return so_array # Earth Coordinates Inertial
    
    # Used to return the last position vector transformation
    def get_last_position_rotation_matrix(self, with_state_column_length = True):
        H_flat = np.array(list(self.csConverter.GetLastRotationMatrix().GetDataVector())).copy() # length 9 vector, row wise flattened
        if with_state_column_length:
            num_sf = len(self.solve_for_states)
            num_states = 6 + num_sf
            H = np.hstack(( H_flat.reshape((3,3)), np.zeros((3, num_states-3)) ))
        else:
            H = H_flat.reshape((3,3))
        return H        

# If you pass in dt, the function assumes the time is in seconds
# If you pass in times, you can specify the time_units
# If you pass in neither dt or times, defaults to "Step k" units
def plot_kf_errors_against_sigma_bound(xs_true, xs_kf, Ps_kf, sigma, dt=None, times=None, time_units=None, solve_for_labels=None):
    es = xs_true - xs_kf
    sigmas = np.array([ sigma * np.diag(P)**0.5 for P in Ps_kf ])

    N = xs_kf.shape[0]
    if (dt is None) and (times is None):
        Ts = np.arange(N)
        x_label = "Step k"
    elif (dt is not None) and (times is None):
        Ts = np.arange(N) * dt / 3600
        x_label = "Time (hours)"
    elif (dt is None) and (times is not None):
        Ts = times 
        x_label = "Time (" + time_units + ")" if time_units is not None else "Time"
    else:
        print("Specify either dt or times, not both! Defaulting to times...")
        Ts = times 
        x_label = "Time (" + time_units + ")" if time_units is not None else "Time"
    pos_labels = ["PosX\n(km)", "PosY\n(km)", "PosZ\n(km"]
    vel_labels = ["VelX\n(km/s)", "VelY\n(km/s)", "VelZ\n(km/s)"]
    plt.figure()
    plt.suptitle("Errors for Positions(x,y,z), Velocities(x,y,z), Solve Fors\nErrors are in Blue, {}-Sigma Bound in Red".format(sigma))
    num_states = xs_kf.shape[1]
    for i in range(num_states):
        plt.subplot(num_states,1,i+1)
        plt.plot(Ts, es[:, i], 'b')
        plt.plot(Ts, sigmas[:,i], 'r')
        plt.plot(Ts, -sigmas[:,i], 'r')
        if i < 3:
            plt.ylabel(pos_labels[i])
        elif i < 6:
            plt.ylabel(vel_labels[i-3])
        elif solve_for_labels is not None:
            plt.ylabel(solve_for_labels[i-6])
    plt.xlabel(x_label)
    plt.show()