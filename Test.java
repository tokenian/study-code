import java.net.InetAddress;
import java.io.IOException;

public class Test {
	public static void main(String[] args) throws IOException{
		InetAddress address = InetAddress.getLocalHost();
		String ip = address.getHostAddress();
		byte[] bs = ip.getBytes();
		System.out.println(ip);
		System.out.println(bs.length);
	}
}
