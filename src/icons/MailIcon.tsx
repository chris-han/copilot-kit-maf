import { Mail as LucideMail, LucideProps } from 'lucide-react';

const MailIcon = ({ className, ...props }: LucideProps) => {
  return <LucideMail className={className} {...props} />;
};

export default MailIcon;