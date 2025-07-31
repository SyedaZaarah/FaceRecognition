using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.VisualBasic;  
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace FaceRecognitionWPF
{
    public partial class MainWindow : Window
    {
        private VideoCapture _capture;
        private DispatcherTimer _timer;
        private CascadeClassifier _faceCascade;
        private LBPHFaceRecognizer _faceRecognizer;
        private Dictionary<int, string> _labelNameMap = new Dictionary<int, string>();
        private int _nextLabel = 0;
        private string _knownFacesDir = "known_faces";
        private string _appointmentsFile = "appointments.json";
        private List<Appointment> _appointments = new List<Appointment>();
        private Mat _frame = new Mat();

        public MainWindow()
        {
            InitializeComponent();

            _faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
            _faceRecognizer = new LBPHFaceRecognizer(1, 8, 8, 8, 100);

            LoadAppointments();
            LoadKnownFaces();

            _capture = new VideoCapture(0);
            _capture.ImageGrabbed += Capture_ImageGrabbed;
            _capture.Start();

            _timer = new DispatcherTimer();
            _timer.Interval = TimeSpan.FromMilliseconds(33);
            _timer.Tick += Timer_Tick;
            _timer.Start();
        }

        private void LoadAppointments()
        {
            if (File.Exists(_appointmentsFile))
            {
                try
                {
                    string json = File.ReadAllText(_appointmentsFile);
                    _appointments = JsonConvert.DeserializeObject<List<Appointment>>(json) ?? new List<Appointment>();
                }
                catch
                {
                    MessageBox.Show("Failed to load appointments.json");
                    _appointments = new List<Appointment>();
                }
            }
            else
            {
                MessageBox.Show("appointments.json file not found.");
            }
        }

        private void LoadKnownFaces()
        {
            if (!Directory.Exists(_knownFacesDir))
            {
                MessageBox.Show($"Known faces folder '{_knownFacesDir}' not found.");
                return;
            }

            var faceImages = new List<Image<Gray, byte>>();
            var faceLabels = new List<int>();

            foreach (var file in Directory.GetFiles(_knownFacesDir, "*.jpg"))
            {
                try
                {
                    var imgColor = new Image<Bgr, byte>(file);
                    var imgGray = imgColor.Convert<Gray, byte>();

                    var faces = _faceCascade.DetectMultiScale(imgGray, 1.1, 4);
                    if (faces.Length == 0)
                        continue;

                    var faceImg = imgGray.Copy(faces[0]).Resize(100, 100, Inter.Linear);
                    faceImages.Add(faceImg);

                    string name = Path.GetFileNameWithoutExtension(file);
                    _labelNameMap[_nextLabel] = name;
                    faceLabels.Add(_nextLabel);

                    _nextLabel++;
                }
                catch { }
            }

            if (faceImages.Count == 0)
            {
                MessageBox.Show("No valid faces found for training.");
                return;
            }

            var trainingImages = new VectorOfMat();
            foreach (var img in faceImages)
            {
                Mat matImg;
                try
                {
                    matImg = img.Mat;
                }
                catch
                {
                    matImg = new Mat(img.Bitmap);
                }
                trainingImages.PushBack(matImg);
            }

            int[] labelsArray = faceLabels.ToArray();
            using (Mat labelMat = new Mat(labelsArray.Length, 1, DepthType.Cv32S, 1))
            {
                Marshal.Copy(labelsArray, 0, labelMat.DataPointer, labelsArray.Length);
                _faceRecognizer.Train(trainingImages, labelMat);
            }
        }

        private void Capture_ImageGrabbed(object sender, EventArgs e)
        {
            if (_capture == null) return;
            _capture.Retrieve(_frame);
        }

        private async void Timer_Tick(object sender, EventArgs e)
        {
            if (_frame.IsEmpty)
                return;

            using var imgGray = _frame.ToImage<Gray, byte>();
            var faces = _faceCascade.DetectMultiScale(imgGray, 1.1, 4);

            foreach (var face in faces)
            {
                CvInvoke.Rectangle(_frame, face, new MCvScalar(0, 255, 0), 2);
                var faceImg = imgGray.Copy(face).Resize(100, 100, Inter.Linear);

                var result = _faceRecognizer.Predict(faceImg);
                string label = "Unknown";

                if (result.Label != -1 && result.Distance < 70 && _labelNameMap.ContainsKey(result.Label))
                {
                    label = _labelNameMap[result.Label];
                }
                else
                {
                    
                    _timer.Stop();
                    _capture.Pause();

                    var hasAppointment = MessageBox.Show("Do you have an appointment?", "Appointment Check", MessageBoxButton.YesNo);

                    if (hasAppointment == MessageBoxResult.No)
                    {
                        string reason = Interaction.InputBox("Why are you here?", "Reason for visit", "");
                        LogReason(reason);
                    }
                    else
                    {
                        string name = Interaction.InputBox("Please enter your name", "Name", "");

                        bool nameFound = _appointments.Any(a => a.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
                        if (!nameFound)
                        {
                            MessageBox.Show("No appointment found for this name.");
                        }
                        else
                        {
                           
                            bool faceMatch = VerifyFaceMatch(faceImg, name);
                            if (faceMatch)
                            {
                                MessageBox.Show("Access Granted");
                                label = name;
                            }
                            else
                            {
                                MessageBox.Show("Access Denied - Face does not match appointment");
                                label = "Access Denied";
                            }
                        }
                    }

                    
                    _capture.Start();
                    _timer.Start();
                }

                CvInvoke.PutText(_frame, label, new System.Drawing.Point(face.X, face.Y - 10),
                    FontFace.HersheySimplex, 0.8, new MCvScalar(255, 0, 0), 2);
            }

            imgWebcam.Source = ToBitmapSource(_frame);
        }

        private bool VerifyFaceMatch(Image<Gray, byte> capturedFace, string name)
        {
            // Load stored face image by name
            string filePath = Path.Combine(_knownFacesDir, $"{name}.jpg");
            if (!File.Exists(filePath)) return false;

            var storedColor = new Image<Bgr, byte>(filePath);
            var storedGray = storedColor.Convert<Gray, byte>();
            var storedFace = DetectSingleFace(storedGray);
            if (storedFace == null) return false;

            var storedFaceImg = storedGray.Copy(storedFace.Value).Resize(100, 100, Inter.Linear);

            // Compare capturedFace with storedFaceImg
            var result = _faceRecognizer.Predict(capturedFace);
            return result.Label != -1 && result.Distance < 70 && _labelNameMap[result.Label] == name;
        }

        private System.Drawing.Rectangle? DetectSingleFace(Image<Gray, byte> imgGray)
        {
            var faces = _faceCascade.DetectMultiScale(imgGray, 1.1, 4);
            if (faces.Length == 1)
                return faces[0];
            return null;
        }

        private void LogReason(string reason)
        {
            if (string.IsNullOrWhiteSpace(reason)) return;

            string logFile = "visitor_reasons.log";
            string logText = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} - Reason: {reason}";
            try
            {
                File.AppendAllLines(logFile, new[] { logText });
            }
            catch { }
        }

        private BitmapSource ToBitmapSource(Mat mat)
        {
            using var bitmap = mat.ToBitmap();
            var hBitmap = bitmap.GetHbitmap();

            try
            {
                var bitmapSource = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                    hBitmap, IntPtr.Zero, System.Windows.Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());
                bitmapSource.Freeze();
                return bitmapSource;
            }
            finally
            {
                DeleteObject(hBitmap);
            }
        }

        [DllImport("gdi32.dll")]
        private static extern bool DeleteObject(IntPtr hObject);

        protected override void OnClosed(EventArgs e)
        {
            _timer.Stop();
            _capture?.Stop();
            _capture?.Dispose();
            base.OnClosed(e);
        }
    }

    public class Appointment
    {
        public string Name { get; set; }
    }
}
