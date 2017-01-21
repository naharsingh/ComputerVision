using System;
using System.Collections.Generic;
using System.Drawing;

namespace MachineLearningOCRTool
{   
    public static class Common
    {
        private static string[] m_letters = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", 
                                              "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", 
                                              "U", "V", "W", "X", "Y", "Z"};

        /*private static string[] m_letters = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};*/

        /*private static string[] m_letters = {"A", "B", "C", "D", "E", "F", "G", "H"};*/


        public static string[] Letters {get { return m_letters; }}

        public static void ForEach<T>(this IEnumerable<T> enumeration, Action<T> action)
        {
            foreach (T item in enumeration)
            {
                action(item);
            }
        }

        public static void SetDoubleBuffered(System.Windows.Forms.Control c)
        {
            System.Reflection.PropertyInfo aProp =
                  typeof(System.Windows.Forms.Control).GetProperty(
                        "DoubleBuffered",
                        System.Reflection.BindingFlags.NonPublic |
                        System.Reflection.BindingFlags.Instance);

            aProp.SetValue(c, true, null);
        }

        public static int GetColorAverage(Color color)
        {
            return (color.R + color.G + color.B)/3;
        }
    }
}
