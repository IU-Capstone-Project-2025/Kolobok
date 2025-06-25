using UnityEngine;

public class CameraOrbitalController : MonoBehaviour
{
    public Transform center;
    public float left_right_max_angle = 10;
    public Vector3 rotationDir = Vector3.up;
    public Vector3 left_right_rotation_bend_axis = Vector3.right;
    public float speed = 1;
    public float minRadius = 1;
    public float maxRadius = 2;
    private Quaternion originalLocalRotation;
    private Vector3 originalLocalPosition;
    private float currentAngle = 0;
    private float orbitAngle = 0f;
private float bendPhase = 0f;
    private void Start()
    {
        originalLocalRotation = transform.localRotation;
        originalLocalPosition = transform.localPosition;
    }


private void Update()
{
    if (center == null) return;

    // 1. Orbit angle increases over time
    orbitAngle += speed * Time.deltaTime;

    // 2. Calculate orbit position with smooth left/right shift
    float radius = Mathf.Lerp(minRadius, maxRadius, 0.5f);
    
    // Calculate the base orbit position
    Vector3 baseOffset = Quaternion.AngleAxis(orbitAngle, rotationDir.normalized) * (Vector3.forward * radius);
    
    // Add smooth left/right shift perpendicular to the orbit plane
    float shiftAmount = Mathf.Sin(orbitAngle) * left_right_max_angle;
    Vector3 shiftOffset = left_right_rotation_bend_axis.normalized * shiftAmount;
    
    // Combine base orbit with shift
    Vector3 finalOffset = baseOffset + shiftOffset;

    // 3. Set camera position
    transform.position = center.position + finalOffset;

    // 4. Look at the center
    transform.LookAt(center.position, Vector3.up);
}
   
    
}
