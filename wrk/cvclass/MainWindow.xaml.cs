using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Navigation;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace cvclass
{
	/// <summary>
	/// An empty window that can be used on its own or navigated to within a Frame.
	/// </summary>
	public sealed partial class MainWindow : Window
	{
		public MainWindow()
		{
			this.InitializeComponent();
		}

		private void Run_Click(object sender, RoutedEventArgs e)
		{
			var pFilepath = "D:\\Home\\Rink\\projects\\assets\\networks\\fcn-resnet50-12-int8.onnx";
			var pImageFilepath = "C:\\Users\\Rink\\OneDrive\\Pictures\\01.jpg";
			if (FCN.Run(pFilepath, pImageFilepath) == false)
			{
				;
			}
		}
	}
}
