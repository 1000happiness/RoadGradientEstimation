import numpy as np

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def solve_AX_eq_YB(A_list, B_list):
    sample_num = len(A_list)
    if sample_num < 3:
        return None, None

    A = np.matrix(np.zeros((12 * sample_num, 24)))
    b = np.matrix(np.zeros((12 * sample_num, 1)))

    for i in range(sample_num):
        A_line, b_line = get_A_and_b(A_list[i], B_list[i])
        A[i * 12: i * 12 + 12,:] = A_line
        b[i * 12: i * 12 + 12,:] = b_line

    x = np.linalg.solve(A, b)
    # x = (A.T * A).I * A.T * b
    R_X = x[ 0: 9].reshape((3,3))
    R_Y = x[ 9:18].reshape((3,3))
    T_X = x[18:21]
    T_Y = x[21:24]

    res_X = np.matrix(np.zeros(4,4))
    res_X[0:3, 0:3] = R_X
    res_X[0:3, 3:4] = T_X
    res_Y = np.matrix(np.zeros(4,4))
    res_Y[0:3, 0:3] = R_Y
    res_Y[0:3, 3:4] = T_Y

    return res_X, res_Y


# in Ax=b
def get_A_and_b(A_4_4: np.ndarray, B_4_4: np.ndarray):
    R_A = np.matrix(A_4_4[0:3,0:3])
    T_A = np.matrix(A_4_4[0:3,3:4])
    R_B = np.matrix(B_4_4[0:3,0:3])
    T_B = np.matrix(B_4_4[0:3,3:4])

    # R_A Xmul I 
    I_3_3 = np.ones((3,3))
    R_A_Xmul_I = np.zeros((9,9))#9*9
    for i in range(0,3):
        for k in range(0,3):
            R_A_Xmul_I[i*3:i*3+3, k*3:k*3+3] = R_A[i,k] * I_3_3

    # I Xmul R_B
    I_Xmul_R_B = np.zeros((9,9))#9*9
    for i in range(0,3):
        for k in range(0,3):
            I_Xmul_R_B[i*3:i*3+3, k*3:k*3+3] = R_B.T

    # I Xmul T_B(T)
    I_Xmul_T_B = np.zeros((3,9))#9*9
    for i in range(0,3):
        for k in range(0,3):
            I_Xmul_T_B[i:i+1, k*3:k*3+3] = T_B.ravel()

    A = np.matrix(np.zeros((12,24)))
    A[ 0: 9,  0: 9] = R_A_Xmul_I
    A[ 0: 9,  9:18] = (-1) * I_Xmul_R_B
    A[ 9:12,  9:18] = I_Xmul_T_B
    A[ 9:12, 18:21] = (-1) * R_A
    A[ 9:12, 21:24] = I_3_3

    b = np.matrix(np.zeros((12,1)))
    b[ 9:12,:] = T_A

    return A, b
