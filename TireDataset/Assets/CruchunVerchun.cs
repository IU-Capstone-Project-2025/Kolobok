using UnityEngine;
using System.Collections.Generic;
using UnityEditor;


public class CruchunVerchun : MonoBehaviour
{
    public Vector3 rotationForward = Vector3.forward;
    public Vector3 rotationLeftRight = Vector3.right;
    public Transform center;
    public Material material;
    public float speed = 1;
    public float max_delta_x_angle = 10;
    public float currentTireThin = 1;

    [System.Serializable]
    public class MaterialDeepControl
    {
        public string paramName;
        public float value;
    }

    [System.Serializable]
    public class DeepConfig
    {
        public float deepness = 0;
        public bool useControls = false;
        // INSERT_YOUR_CODE
#if UNITY_EDITOR
        [UnityEditor.CustomPropertyDrawer(typeof(DeepConfig))]
        public class DeepConfigDrawer : UnityEditor.PropertyDrawer
        {
            public override void OnGUI(UnityEngine.Rect position, UnityEditor.SerializedProperty property, UnityEngine.GUIContent label)
            {
                UnityEditor.EditorGUI.BeginProperty(position, label, property);

                var deepnessProp = property.FindPropertyRelative("deepness");
                var useControlsProp = property.FindPropertyRelative("useControls");
                var controlsProp = property.FindPropertyRelative("materialDeepControls");

                float y = position.y;
                float lineHeight = UnityEditor.EditorGUIUtility.singleLineHeight;
                float spacing = UnityEditor.EditorGUIUtility.standardVerticalSpacing;

                // Draw deepness
                UnityEditor.EditorGUI.PropertyField(
                    new UnityEngine.Rect(position.x, y, position.width, lineHeight),
                    deepnessProp);
                y += lineHeight + spacing;

                // Draw useControls
                UnityEditor.EditorGUI.PropertyField(
                    new UnityEngine.Rect(position.x, y, position.width, lineHeight),
                    useControlsProp);
                y += lineHeight + spacing;

                // Draw controls list if useControls is true
                if (useControlsProp.boolValue)
                {
                    UnityEditor.EditorGUI.PropertyField(
                        new UnityEngine.Rect(position.x, y, position.width, UnityEditor.EditorGUI.GetPropertyHeight(controlsProp, true)),
                        controlsProp, true);
                }

                UnityEditor.EditorGUI.EndProperty();
            }

            public override float GetPropertyHeight(UnityEditor.SerializedProperty property, UnityEngine.GUIContent label)
            {
                float height = 0f;
                var deepnessProp = property.FindPropertyRelative("deepness");
                var useControlsProp = property.FindPropertyRelative("useControls");
                var controlsProp = property.FindPropertyRelative("materialDeepControls");

                height += UnityEditor.EditorGUIUtility.singleLineHeight + UnityEditor.EditorGUIUtility.standardVerticalSpacing; // deepness
                height += UnityEditor.EditorGUIUtility.singleLineHeight + UnityEditor.EditorGUIUtility.standardVerticalSpacing; // useControls

                if (useControlsProp.boolValue)
                {
                    height += UnityEditor.EditorGUI.GetPropertyHeight(controlsProp, true);
                }

                return height;
            }
        }
#endif
        public List<MaterialDeepControl> materialDeepControls = new List<MaterialDeepControl>();
    }

    public List<DeepConfig> deepConfigs = new List<DeepConfig>();
    private Quaternion originalLocalRotation;
 void Start()
   {
       originalLocalRotation = transform.localRotation;
   }
   
    void Update()
   {
       // Calculate the spin (rolling) around rotationForward
       float spinAngle = speed * Time.time;
       Quaternion spinRotation = Quaternion.AngleAxis(spinAngle, rotationForward.normalized);

       // Calculate the bend (oscillation) around rotationLeftRight
       float bendAngle = Mathf.Sin(Time.time * speed) * max_delta_x_angle;
       Quaternion bendRotation = Quaternion.AngleAxis(bendAngle, rotationLeftRight.normalized);

       // Combine: bend, then spin, then original
       transform.localRotation = bendRotation * spinRotation * originalLocalRotation;
   }
}