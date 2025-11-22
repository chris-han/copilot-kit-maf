import { MessageCircle as LucideMessageCircle, LucideProps } from 'lucide-react';

const Chat = ({ className, ...props }: LucideProps) => {
  return <LucideMessageCircle className={className} {...props} />;
};

export default Chat;