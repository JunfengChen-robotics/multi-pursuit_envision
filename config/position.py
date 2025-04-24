import numpy as np

class InitPos():
    
    @classmethod
    def get_predefined_positions(cls, case_number):
    
        if case_number == "case5":
            
            
            # ++++++++++++++++++++++++++++
            # big scenario
            # predefined_positions = [np.array([6.0, 0.5]),np.array([7.5,2]), np.array([3, 6])]        
            
            # ++++++++++++++++++++++++++++

            # +++++++++++++++++++++++++++++
            # small scenario
            # predefined_positions = [np.array([2.0, 2.7]),np.array([5,0.5]), np.array([2.0,1.5])]
            # predefined_positions = [np.array([0.5, 4.0]),np.array([4,6.0]), np.array([0.5,7.5])]
            
            # predefined_positions = [np.array([3, 3.5]),np.array([6,1.5]), np.array([2.5,6.5])]
            # predefined_positions = [np.array([3, 3.5]),np.array([6,0.5]), np.array([6.5,3])]
            # predefined_positions = [np.array([1.5, 7.5]),np.array([1,1]), np.array([3,5.5])]
            # +++++++++++++++++++++++++++++
            
            
            # predefined_positions = [np.array([3, 3.5]),np.array([6,0.5]),np.array([6, 6.5]), np.array([2.5,1.5])]
            # predefined_positions = [np.array([3, 3.5]),np.array([6,0.5]),np.array([6, 6.5]), np.array([2.5,6.5])]
            # predefined_positions = [np.array([3, 3.5]),np.array([6,0.5]),np.array([6, 6.5]), np.array([6.5,3])]
            predefined_positions = [np.array([3.9, 0.5]),np.array([4.0,0.5]), np.array([4.1,0.5]), np.array([3.5,7.5])]
            return predefined_positions

        elif case_number == "case6":
            
            # ---------------------- pursuers vs. evader (2:1) ----------------------
            # predefined_positions = {"pos":[np.array([2, 1.8]),np.array([2,2]), np.array([5,9.5])],"label":"init1"}
            # predefined_positions = {"pos":[np.array([9, 9.5]),np.array([8.8,9.5]), np.array([5,0.5])],"label":"init2"}
            # predefined_positions = {"pos":[np.array([5, 9.5]),np.array([5,9.0]), np.array([5,5])],"label":"init3"}
            # predefined_positions = {"pos":[np.array([0.5,0.5]),np.array([9.5,0.5]), np.array([5,5])],"label":"init4"}
            # predefined_positions = {"pos":[np.array([5, 6]),np.array([5,5.5]), np.array([5,5]), np.array([9.5,5])],"label":"init5"}
            # predefined_positions = {"pos":[np.array([2, 1.8]),np.array([2,2]), np.array([5,9.5])],"label":"test"}

            
            # ---------------------- pursuers vs. evader 3:1）------------------------
            predefined_positions = [np.array([2, 1.8]),np.array([2,2]), np.array([2,2.2]), np.array([5,9.5])]
            # predefined_positions = [np.array([9, 9.5]),np.array([8.8,9.5]), np.array([9.5,5]), np.array([5,0.5])]
            # predefined_positions = [np.array([5, 9.5]),np.array([5,9.0]), np.array([5,8.5]), np.array([5,5])]
            # predefined_positions = [np.array([0.5,0.5]),np.array([9.5,0.5]), np.array([9.5,9.5]), np.array([5,5])]
            # predefined_positions = [np.array([5, 6]),np.array([5,5.5]), np.array([5,5]), np.array([9.5,5])]
            # predefined_positions = [np.array([2, 1.8]),np.array([2,2]), np.array([2,2.2]), np.array([5,9.5])


            # ---------------------- pursuers vs. evader (4:1) ----------------------
            # predefined_positions = [np.array([2, 1.8]),np.array([2,2]), np.array([2,2.2]), np.array([2,2.4]), np.array([5,9.5])]
            # predefined_positions = [np.array([9, 9.5]),np.array([8.8,9.5]), np.array([9.5,5]), np.array([9.2,9.5]), np.array([5,0.5])]
            # predefined_positions = [np.array([5, 9.5]),np.array([5,9.0]), np.array([5,8.5]), np.array([5,8.0]), np.array([5,5])]
            # predefined_positions = [np.array([0.5,0.5]),np.array([9.5,0.5]), np.array([9.5,9.5]), np.array([0.5,9.5]), np.array([5,5])]
            # predefined_positions = [np.array([5, 6]),np.array([5,5.5]), np.array([5,5]), np.array([5,4.5]), np.array([9.5,5])]
            # predefined_positions = [np.array([2, 1.8]),np.array([2,2]), np.array([2,2.2]), np.array([2,2.4]), np.array([5,9.5])]

            # ---------------------- pursuers vs. evader (5:1) ----------------------

            # predefined_positions = [np.array([2, 1.8]),np.array([2,2]), np.array([2,2.2]), np.array([2,2.4]),  np.array([2,2.1]), np.array([5,9.5])]
            # predefined_positions = [np.array([9, 9.5]),np.array([8.8,9.5]), np.array([9.5,5]), np.array([9.2,9.5]), np.array([8.2,9.5]), np.array([5,0.5])]
            # predefined_positions = [np.array([5, 9.5]),np.array([5,9.0]), np.array([5,8.5]), np.array([5,8.0]),  np.array([5,7.5]), np.array([5,5])]
            # predefined_positions = [np.array([0.5,0.5]),np.array([9.5,0.5]), np.array([9.5,9.5]), np.array([0.5,9.5]),  np.array([0.5,5]), np.array([5,5])]
            # predefined_positions = [np.array([5, 6]),np.array([5,5.5]), np.array([5,5]), np.array([5,4.5]),  np.array([5,4]), np.array([9.5,5])]
            return predefined_positions

        elif case_number == "case10":
            # predefined_positions = [np.array([3, 3.5]),np.array([6,0.5]),np.array([0.5,4]), np.array([2.5,1.5])]
            # predefined_positions = [np.array([6.5, 4]),np.array([9,6]), np.array([8,9.5]), np.array([7,8])]
            # predefined_positions = [np.array([2, 1.8]),np.array([5,1]), np.array([9.5,2]), np.array([5,9.5])]
            # predefined_positions = [np.array([2, 1.8]),np.array([2,1.6]), np.array([2,1.4]), np.array([5,9.5])]
            # predefined_positions = [np.array([1.8, 2]),np.array([8,3.8]),np.array([8,9.5]), np.array([6,6])]
            predefined_positions = [np.array([8, 1.0]),np.array([8,0.5]), np.array([8,0.8]), np.array([1.5,9.5])]
            # predefined_positions = [np.array([2.5, 1.0]),np.array([2.2,1.1]), np.array([2.6,1.1]), np.array([9.5,9.5])]
            # predefined_positions = [np.array([3.8, 0.3]),np.array([4.2,0.3]), np.array([4.6,0.3]), np.array([5,9.5])] 
            # predefined_positions = [np.array([6, 0.5]),np.array([6.2,0.5]), np.array([5.8,0.5]), np.array([1,9.5])]
            return predefined_positions
            